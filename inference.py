import os
import sys
import argparse
import torch
import json
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO

# 통합 모듈 임포트
from roi_downsampler import IntegratedROIDownsamplerNetwork

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='통합 ROI 다운샘플러 추론')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='입력 이미지 시퀀스가 포함된 디렉토리')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='결과 이미지가 저장될 디렉토리')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='ROI 다운샘플러 모델 체크포인트 경로')
    
    parser.add_argument('--seg_model_path', type=str, default='models/yolov8n-seg.pt',
                        help='세그멘테이션 모델 경로')
    
    parser.add_argument('--scale_factor', type=int, default=2,
                        help='다운샘플링 비율 (2=1/2, 4=1/4)')
    
    parser.add_argument('--max_size', type=int, default=None,
                        help='이미지의 최대 크기 (None=제한 없음)')
    
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='마스크 생성 임계값')
    
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드 활성화 (추가 로깅 및 시각화)')
    
    return parser.parse_args()

def load_models(args, device):
    """모델 로드"""
    print("모델 로드 중...")
    
    # 세그멘테이션 모델 로드
    try:
        seg_model = YOLO(args.seg_model_path)
        print(f"세그멘테이션 모델 로드됨: {args.seg_model_path}")
    except Exception as e:
        print(f"세그멘테이션 모델 로드 실패: {e}")
        sys.exit(1)
    
    # ROI 다운샘플러 모델 로드
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # 모델 구성 매개변수 추출
        if 'config' in checkpoint:
            config = checkpoint['config']
            scale_factor = config.get('scale_factor', args.scale_factor)
            mask_threshold = config.get('mask_threshold', args.mask_threshold)
            bg_color = config.get('bg_color', [0.5, 0.5, 0.5])
        else:
            # 이전 버전 호환성
            scale_factor = args.scale_factor
            mask_threshold = args.mask_threshold
            bg_color = [0.5, 0.5, 0.5]
        
        # 통합 ROI 다운샘플러 네트워크 초기화
        roi_model = IntegratedROIDownsamplerNetwork(
            scale_factor=scale_factor,
            in_channels=3,
            features=64,
            bg_color=bg_color,
            mask_threshold=mask_threshold
        )
        
        # 모델 가중치 로드
        if 'model_state_dict' in checkpoint:
            roi_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 이전 버전 호환성
            roi_model.load_state_dict(checkpoint)
        
        roi_model.eval()
        roi_model.to(device)
        
        print(f"ROI 다운샘플러 모델 로드됨: {args.checkpoint_path}")
        print(f"스케일 팩터: {scale_factor}x")
        
        return seg_model, roi_model
    except Exception as e:
        print(f"ROI 다운샘플러 모델 로드 실패: {e}")
        sys.exit(1)

def find_image_files(directory, recursive=True):
    """디렉토리에서 이미지 파일 찾기"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    
    if recursive:
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(directory, '**', ext), recursive=True))
            # 대문자 확장자도 검색
            image_files.extend(glob(os.path.join(directory, '**', ext.upper()), recursive=True))
    else:
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(directory, ext)))
            # 대문자 확장자도 검색
            image_files.extend(glob(os.path.join(directory, ext.upper())))
    
    return sorted(image_files)

def get_sequence_folders(input_dir):
    """시퀀스 폴더 목록 가져오기"""
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    return sorted(folders)

def process_sequence(seq_folder, input_dir, output_dir, seg_model, roi_model, args, device):
    """시퀀스 처리"""
    seq_path = os.path.join(input_dir, seq_folder)
    output_seq_path = os.path.join(output_dir, seq_folder)
    os.makedirs(output_seq_path, exist_ok=True)
    
    # 이미지 파일 찾기
    image_files = find_image_files(seq_path)
    
    if not image_files:
        print(f"경고: {seq_path}에서 이미지 파일을 찾을 수 없습니다.")
        return
    
    # 파일 정렬 (숫자 순서로)
    def get_frame_number(path):
        # 파일명에서 숫자 추출 (예: frame_001.jpg -> 1)
        filename = os.path.basename(path)
        try:
            # '_' 또는 '.' 구분자 다음의 숫자 추출
            if '_' in filename:
                num_part = filename.split('_')[-1].split('.')[0]
            else:
                num_part = filename.split('.')[0]
            
            # 숫자가 아닌 문자 제거
            num = ''.join(c for c in num_part if c.isdigit())
            return int(num) if num else 0
        except:
            return 0
    
    image_files.sort(key=get_frame_number)
    print(f"처리할 이미지: {len(image_files)}개")
    
    # 시퀀스 처리 준비
    # 시간적 메모리 초기화
    roi_model.reset_temporal_memory()
    prev_frame = None
    
    # 시퀀스의 모든 이미지 처리
    for i, img_path in enumerate(tqdm(image_files, desc=f"시퀀스 처리 중: {seq_folder}")):
        # 이미지 로드
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"경고: 이미지를 로드할 수 없습니다: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 제한 (필요한 경우)
        if args.max_size is not None:
            h, w = img.shape[:2]
            if max(h, w) > args.max_size:
                scale = args.max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                img_rgb = cv2.resize(img_rgb, new_size)
        
        # 객체 탐지 및 세그멘테이션
        results = seg_model.predict(
            source=img_rgb, 
            conf=0.25,  # 기본 신뢰도 임계값
            verbose=False,
            save=False
        )[0]
        
        # 마스크 및 ROI 다운샘플링 처리
        with torch.no_grad():
            # 이미지를 텐서로 변환
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor / 255.0
            img_tensor = img_tensor.to(device)
            
            # 마스크 생성
            object_mask = None
            if hasattr(results, 'masks') and results.masks is not None and len(results.masks.data) > 0:
                masks = []
                for mask_data in results.masks.data:
                    mask = mask_data.cpu().numpy()
                    # 마스크 크기 조정
                    if mask.shape[:2] != img_rgb.shape[:2]:
                        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
                    masks.append(mask)
                
                # 마스크 결합
                if masks:
                    combined_mask = np.zeros_like(masks[0])
                    for mask in masks:
                        combined_mask = np.maximum(combined_mask, mask)
                    
                    # 마스크를 텐서로 변환
                    mask_tensor = torch.from_numpy(combined_mask).float().unsqueeze(0).unsqueeze(0)
                    mask_tensor = mask_tensor.to(device)
                    object_mask = mask_tensor
            
            # 이전 프레임 준비
            if prev_frame is not None:
                prev_frame_tensor = torch.from_numpy(prev_frame).float().permute(2, 0, 1).unsqueeze(0)
                prev_frame_tensor = prev_frame_tensor / 255.0
                prev_frame_tensor = prev_frame_tensor.to(device)
            else:
                prev_frame_tensor = None
            
            # ROI 다운샘플링 실행 (추론 모드)
            downsampled, mask = roi_model.inference(img_tensor, object_mask, prev_frame_tensor)
            
            # 후처리
            downsampled = downsampled.squeeze(0).permute(1, 2, 0)
            downsampled = (downsampled * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
            downsampled = cv2.cvtColor(downsampled, cv2.COLOR_RGB2BGR)
        
        # 현재 프레임 저장 (다음 반복을 위한 이전 프레임으로)
        prev_frame = img_rgb.copy()
        
        # 결과 저장
        output_filename = os.path.basename(img_path)
        output_path = os.path.join(output_seq_path, output_filename)
        cv2.imwrite(output_path, downsampled)
        
        # 디버그 모드에서 시각화
        if args.debug and i % 10 == 0:  # 10프레임마다 시각화
            # 마스크 시각화
            if mask is not None:
                mask_vis = mask.squeeze().cpu().numpy()
                mask_vis = (mask_vis * 255).astype(np.uint8)
                mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
                mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
                alpha = 0.5
                mask_overlay = cv2.addWeighted(img, 1-alpha, cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR), alpha, 0)
            else:
                mask_overlay = img.copy()
            
            # 원본, 마스크, 결과 이미지 병합
            h, w = img.shape[:2]
            dh, dw = downsampled.shape[:2]
            
            # 크기 일치시키기
            downsampled_resized = cv2.resize(downsampled, (w, h))
            
            # 이미지 가로로 병합
            combined = np.hstack((img, mask_overlay, downsampled_resized))
            
            # 디버그 이미지 저장
            debug_dir = os.path.join(output_seq_path, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"debug_{output_filename}")
            cv2.imwrite(debug_path, combined)
    
    print(f"시퀀스 처리 완료: {seq_folder}")

def main():
    """메인 함수"""
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 모델 로드
    seg_model, roi_model = load_models(args, device)
    
    # 시퀀스 폴더 목록 가져오기
    sequence_folders = get_sequence_folders(args.input_dir)
    print(f"처리할 시퀀스 폴더: {len(sequence_folders)}개")
    
    if not sequence_folders:
        print(f"경고: {args.input_dir}에서 시퀀스 폴더를 찾을 수 없습니다.")
        sys.exit(1)
    
    # 각 시퀀스 처리
    for seq_folder in sequence_folders:
        process_sequence(
            seq_folder=seq_folder,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            seg_model=seg_model,
            roi_model=roi_model,
            args=args,
            device=device
        )
    
    print(f"모든 시퀀스 처리 완료. 결과가 {args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main() 