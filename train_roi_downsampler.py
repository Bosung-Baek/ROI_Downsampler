import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import random
from torchvision.utils import make_grid, save_image
from ultralytics import YOLO
import torch.nn.functional as F
import json
import logging
import time
from datetime import datetime
import glob

from roi_downsampler import IntegratedROIDownsamplerNetwork, train_step
from blip_clip_loss import IntegratedLoss

# NumPy 및 Torch 데이터 타입을 JSON으로 변환하기 위한 사용자 정의 인코더
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        if hasattr(obj, 'item'):  # torch.Tensor의 스칼라 값
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

# 로깅 설정
def setup_logger(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일 이름 설정 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # 로거 설정
    logger = logging.getLogger("roi_downsampler")
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_clip_model(device="cuda", model_name="ViT-B/32"):
    """CLIP 모델 로드 함수"""
    try:
        import clip
        
        print(f"CLIP 모델 로드 시도: {model_name}")
        
        # device를 문자열로 변환하여 안전하게 처리
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = str(device)
        
        # CUDA 사용 가능 여부에 따라 장치 설정
        device_str = "cuda" if torch.cuda.is_available() and "cuda" in device_str else "cpu"
        
        # CLIP 모델 로드
        model, preprocess = clip.load(model_name, device=device_str)
        
        # 평가 모드 설정
        model.eval()
        
        print("CLIP 모델 로드 성공")
        return model, preprocess
        
    except Exception as e:
        print(f"CLIP 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 더미 CLIP 모델 및 전처리 함수 반환
        class DummyModel:
            def __init__(self): pass
            def to(self, *args, **kwargs): return self
            def eval(self): pass
            def encode_image(self, x): return torch.ones(x.shape[0], 512)
            def encode_text(self, x): return torch.ones(x.shape[0], 512)
            
        def dummy_preprocess(x): 
            return torch.zeros(3, 224, 224)
            
        print("더미 CLIP 모델 사용")
        return DummyModel(), dummy_preprocess

def load_blip_model(device="cuda", model_name="Salesforce/blip2-opt-2.7b", hf_token=None):
    """BLIP2 모델 로드 함수"""
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import os
        
        # 환경 변수 설정
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
        print(f"BLIP2 모델 로드 시도: {model_name}")
        
        # 프로세서 로드
        processor = Blip2Processor.from_pretrained(model_name)
        
        # device를 문자열로 변환하여 안전하게 처리
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = str(device)
            
        # CUDA 사용 가능 여부에 따른 dtype 설정
        use_cuda = torch.cuda.is_available() and "cuda" in device_str
        
        # 모델 로드
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_cuda else torch.float32
        )
        
        # 모델을 장치로 이동
        model = model.to(device)
        
        # 평가 모드 설정
        model.eval()
        
        print("BLIP2 모델 로드 성공")
        return processor, model
        
    except Exception as e:
        print(f"BLIP2 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 더미 모델 반환 (간소화 버전)
        class DummyProcessor:
            def __call__(self, images=None, text=None, return_tensors=None):
                return {"input_ids": torch.zeros(1, 10).long(), "attention_mask": torch.zeros(1, 10).long()}
            def batch_decode(self, *args, **kwargs):
                return ["이미지 설명 불가 (더미 모델)"]
                
        class DummyModel:
            def __init__(self): pass
            def to(self, *args, **kwargs): return self
            def eval(self): pass
            def generate(self, *args, **kwargs): 
                return torch.zeros(1, 10).long()
                
        print("더미 BLIP 모델 사용")
        return DummyProcessor(), DummyModel()

def frame_collate_fn(batch):
    """가변 크기 이미지를 배치 처리하기 위한 사용자 정의 Collate 함수"""
    images = []
    paths = []
    
    for item in batch:
        images.append(item["image"])
        paths.append(item["path"])
    
    # 동적 패딩을 위해 각 이미지의 크기 확인
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    
    # 배치 내에서 동일한 크기로 패딩
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        
        # 이미지 패딩 (오른쪽과 아래에 패딩 추가)
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        padded_images.append(img)
    
    # 텐서로 변환하여 배치 생성
    padded_images = torch.stack(padded_images)
    
    return {"image": padded_images, "path": paths, "original_sizes": [(img.shape[1], img.shape[2]) for img in images]}

def sequence_collate_fn(batch):
    """시퀀스 데이터를 배치 처리하기 위한 콜레이트 함수"""
    images = []
    sequences = []
    paths = []
    current_paths = []
    valid_frames_counts = []
    original_sizes = []
    
    for item in batch:
        images.append(item["image"])
        sequences.append(item["sequence"])
        paths.append(item["paths"])
        current_paths.append(item["current_path"])
        valid_frames_counts.append(item["valid_frames"])
        original_sizes.append(item["original_size"])
    
    # 텐서로 변환
    images = torch.stack(images)
    sequences = torch.stack(sequences)
    
    return {
        "image": images,
        "sequence": sequences,
        "paths": paths,
        "current_path": current_paths,
        "valid_frames": valid_frames_counts,
        "original_size": original_sizes
    }

def save_samples(original, downsampled, mask, save_path, nrow=4, compare_masks=False):
    """
    샘플 이미지 저장 (디버깅 및 성능 평가용)
    
    Args:
        original: 원본 이미지 텐서 [B, C, H, W]
        downsampled: 다운샘플링된 이미지 텐서 [B, C, H', W']
        mask: 마스크 텐서 [B, 1, H, W]
        save_path: 저장 경로
        nrow: 행당 이미지 수
        compare_masks: 마스크 비교 모드 활성화 여부
    """
    try:
        # 텐서가 분리되어 있는지 확인 (학습 중 그래디언트 방지하기 위해)
        original = original.detach().cpu()
        downsampled = downsampled.detach().cpu()
        if mask is not None:
            mask = mask.detach().cpu()
        
        # 배치 크기
        batch_size = original.size(0)
        
        # 원본 이미지와 다운샘플링된 이미지의 크기 확인
        _, _, orig_h, orig_w = original.size()
        _, _, down_h, down_w = downsampled.size()
        
        # 모든 이미지에 사용할 표준 크기 선택 (원본 크기 사용)
        target_h, target_w = orig_h, orig_w
        
        # 원본 이미지를 다운샘플링된 크기로 스케일 조정 (비교용)
        # 다운샘플링 크기가 항상 짝수가 되도록 조정
        adjusted_down_h = down_h + (1 if down_h % 2 == 1 else 0)
        adjusted_down_w = down_w + (1 if down_w % 2 == 1 else 0)
        
        original_downscaled = F.interpolate(
            original, 
            size=(adjusted_down_h, adjusted_down_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 다운샘플링된 이미지를 원본 크기로 업샘플링 (비교용)
        # 원본 크기가 항상 짝수가 되도록 조정
        adjusted_orig_h = orig_h + (1 if orig_h % 2 == 1 else 0)
        adjusted_orig_w = orig_w + (1 if orig_w % 2 == 1 else 0)
        
        downsampled_upscaled = F.interpolate(
            downsampled, 
            size=(adjusted_orig_h, adjusted_orig_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 마스크 비교 모드
        if compare_masks and mask is not None:
            # 마스크 크기 조정 및 처리
            try:
                # 공통 크기로 조정 (원본 크기 사용)
                common_size = (adjusted_orig_h, adjusted_orig_w)
                
                # 원본 이미지 확인 및 조정
                if original.shape[2:] != common_size:
                    original = F.interpolate(
                        original, 
                        size=common_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 다운샘플링된 이미지 확인 및 크기 조정
                if downsampled_upscaled.shape[2:] != common_size:
                    downsampled_upscaled = F.interpolate(
                        downsampled_upscaled, 
                        size=common_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 마스크 조정 (1채널 -> 3채널 RGB)
                mask_display = F.interpolate(
                    mask, 
                    size=common_size, 
                    mode='nearest'
                )
                
                # 마스크를 RGB 컬러맵으로 변환 (파란색)
                mask_rgb = torch.cat([
                    torch.zeros_like(mask_display), 
                    torch.zeros_like(mask_display), 
                    mask_display
                ], dim=1)
                
                # 원본 + 마스크 오버레이
                alpha = 0.7
                overlay = original.clone()
                
                # 크기 확인
                if overlay.shape[2:] != mask_rgb.shape[2:]:
                    overlay = F.interpolate(
                        overlay, 
                        size=common_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                overlay = overlay * (1 - alpha) + mask_rgb * alpha
                
                # 최종 이미지 그리드 구성
                grid_tensors = []
                for i in range(min(batch_size, 4)):  # 최대 4개 샘플만 표시
                    # 모든 이미지에 동일한 크기 확인
                    row_images = [
                        original[i],
                        downsampled_upscaled[i],
                        original_downscaled[i],
                        overlay[i]
                    ]
                    
                    # 각 이미지 크기 출력 (디버깅)
                    print(f"그리드 이미지 크기:")
                    for j, img in enumerate(row_images):
                        print(f"  이미지 {j}: {img.shape}")
                    
                    # 크기 일치 확인 
                    for j in range(len(row_images)):
                        # 이미지 크기가 다르다면 공통 크기로 조정
                        if row_images[j].shape[1:] != common_size:
                            row_images[j] = F.interpolate(
                                row_images[j].unsqueeze(0),
                                size=common_size,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)
                    
                    grid_tensors.extend(row_images)
                
                # 이미지 그리드 생성 및 저장
                grid = make_grid(grid_tensors, nrow=4, padding=2, normalize=False)
                save_image(grid, save_path)
                
            except Exception as e:
                print(f"마스크 비교 모드 오류: {e}")
                import traceback
                traceback.print_exc()
                
                # 오류 발생 시 기본 모드로 폴백
                grid_tensors = []
                common_size = (adjusted_orig_h, adjusted_orig_w)
                
                for i in range(batch_size):
                    try:
                        # 이미지 크기 통일
                        orig_img = F.interpolate(
                            original[i].unsqueeze(0),
                            size=common_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        
                        down_img = F.interpolate(
                            downsampled_upscaled[i].unsqueeze(0),
                            size=common_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        
                        grid_tensors.extend([orig_img, down_img])
                    except Exception as e:
                        print(f"이미지 {i} 처리 중 오류: {e}")
                
                # 이미지 그리드 생성 및 저장
                grid = make_grid(grid_tensors, nrow=nrow, padding=2, normalize=False)
                save_image(grid, save_path)
        else:
            # 기본 모드: 원본 / 다운샘플링 비교만
            grid_tensors = []
            common_size = (adjusted_orig_h, adjusted_orig_w)
            
            for i in range(batch_size):
                try:
                    # 이미지 크기 통일
                    orig_img = F.interpolate(
                        original[i].unsqueeze(0),
                        size=common_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    
                    down_img = F.interpolate(
                        downsampled_upscaled[i].unsqueeze(0),
                        size=common_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    
                    grid_tensors.extend([orig_img, down_img])
                except Exception as e:
                    print(f"이미지 {i} 처리 중 오류: {e}")
            
            # 이미지 그리드 생성 및 저장
            grid = make_grid(grid_tensors, nrow=nrow, padding=2, normalize=False)
            save_image(grid, save_path)
    
    except Exception as e:
        print(f"샘플 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def create_transform(max_size, keep_aspect_ratio=True):
    """
    이미지 변환 함수 생성
    
    Args:
        max_size: 최대 이미지 크기
        keep_aspect_ratio: 가로세로 비율 유지 여부
        
    Returns:
        torchvision transforms Compose 객체
    """
    if keep_aspect_ratio:
        # 가로세로 비율 유지하면서 변환
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        # 고정 크기로 변환
        return transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
        ])

def train(
    dataset_path,
    output_dir="output",
    epochs=15,
    batch_size=4,
    learning_rate=1e-5,
    weight_decay=1e-5,
    scale_factor=2,
    max_size=512,
    perceptual_weight=1.0,
    roi_weight=5.0,
    temporal_weight=2.0,
    mask_threshold=0.25,
    compare_masks=True,
    keep_aspect_ratio=True,
    clip_version="ViT-B/32",
    blip_version="Salesforce/blip2-opt-2.7b",
    yolo_version="yolov8n-seg.pt",
    bg_color=None,
    hf_token=None,
    use_sequences=True,
    sequence_length=5,
    sequence_stride=2,
    max_frames_per_sequence=100
):
    """
    통합 ROI 다운샘플러 모델 학습 함수
    
    Args:
        dataset_path: 데이터셋 경로
        output_dir: 출력 디렉토리
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        weight_decay: 가중치 감쇠
        scale_factor: 다운샘플링 배율
        max_size: 최대 이미지 크기
        perceptual_weight: 지각적 손실 가중치 (BLIP/CLIP 등)
        roi_weight: ROI 손실 가중치
        temporal_weight: 시간적 일관성 손실 가중치
        mask_threshold: 마스크 임계값
        compare_masks: 마스크 비교 모드 활성화 여부
        keep_aspect_ratio: 종횡비 유지 여부
        clip_version: CLIP 모델 버전
        blip_version: BLIP 모델 버전
        yolo_version: YOLO 세그멘테이션 모델 버전
        bg_color: 배경 색상 (R,G,B)
        hf_token: Hugging Face 토큰
        use_sequences: 시퀀스 데이터셋 사용 여부
        sequence_length: 시퀀스 길이
        sequence_stride: 시퀀스 스트라이드
        max_frames_per_sequence: 각 시퀀스 폴더에서 최대로 사용할 프레임 수
    
    Returns:
        학습된 모델과 학습 통계
    """
    # 로거 설정
    logger = setup_logger(output_dir)
    logger.info("===== 통합 ROI Downsampler 학습 시작 =====")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"훈련 장치: {device}")
    
    # 모델 경로와 설정을 JSON으로 저장
    config = {
        "scale_factor": scale_factor,
        "max_size": max_size,
        "perceptual_weight": perceptual_weight,
        "roi_weight": roi_weight,
        "temporal_weight": temporal_weight,
        "mask_threshold": mask_threshold,
        "clip_version": clip_version,
        "blip_version": blip_version,
        "yolo_version": yolo_version
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)
    
    # 객체 탐지 모델 로드 (마스킹용 세그멘테이션 모델)
    logger.info(f"YOLO 세그멘테이션 모델 로딩: {yolo_version}")
    try:
        segmentation_model = YOLO(yolo_version)
        logger.info(f"YOLO 세그멘테이션 모델 로드 완료: {yolo_version}")
    except Exception as e:
        logger.error(f"세그멘테이션 모델 로드 실패: {e}")
        segmentation_model = None
    
    # CLIP 모델 로드
    logger.info(f"CLIP 모델 로딩: {clip_version}")
    clip_model, clip_preprocess = load_clip_model(device, clip_version)
    
    # BLIP 모델 로드
    logger.info(f"BLIP 모델 로딩: {blip_version}")
    blip_processor, blip_model = load_blip_model(device, blip_version, hf_token)
    
    # 배경 색상 설정 (기본값: 회색)
    if bg_color is None:
        bg_color = [0.5, 0.5, 0.5]
    
    # 데이터셋 및 로더 생성
    transform = create_transform(max_size, keep_aspect_ratio)
    
    # 데이터셋 및 데이터로더 생성
    if use_sequences:
        # 시퀀스 기반 데이터셋 생성
        dataset = ImagenetSequenceDataset(
            root_dir=dataset_path,
            transform=transform,
            max_size=max_size,
            sequence_length=sequence_length,
            keep_aspect_ratio=keep_aspect_ratio,
            sequence_stride=sequence_stride,
            max_frames_per_sequence=max_frames_per_sequence
        )
        # 시퀀스 데이터 처리를 위한 시퀀스 콜레이트 함수 사용
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,  # 시퀀스 순서 유지
            num_workers=2,
            collate_fn=sequence_collate_fn
        )
    else:
        # 기존 이미지 기반 데이터셋 사용
        dataset = ImagenetFrameDataset(
            root_dir=dataset_path,
            transform=transform,
            max_size=max_size,
            max_seq_len=None,
            keep_aspect_ratio=keep_aspect_ratio
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            collate_fn=frame_collate_fn
        )
    
    logger.info(f"데이터셋 로드 완료: {len(dataset)} 이미지")
    
    # 통합 ROI 다운샘플러 모델 생성
    model = IntegratedROIDownsamplerNetwork(
        scale_factor=scale_factor,
        in_channels=3,
        features=64,
        bg_color=bg_color,
        mask_threshold=mask_threshold
    ).to(device)
    
    # 손실 함수 설정
    criterion = IntegratedLoss(
        perceptual_weight=perceptual_weight,
        roi_weight=roi_weight,
        temporal_weight=temporal_weight,
        clip_model=clip_model,
        blip_processor=blip_processor,
        blip_model=blip_model,
        device=device
    )
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.1)
    
    # 통계 저장용 딕셔너리
    stats = {
        "epoch": [],
        "loss": [],
        "perceptual_loss": [],
        "roi_loss": [],
        "temporal_loss": [],
        "lr": []
    }
    
    # 체크포인트 저장 함수
    def save_checkpoint(epoch, loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config
        }
        torch.save(checkpoint, os.path.join(output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth"))
        # 최근 체크포인트 복사
        torch.save(checkpoint, os.path.join(output_dir, "checkpoints", "latest.pth"))
    
    # 학습 루프
    for epoch in range(epochs):
        logger.info(f"===== 에폭 {epoch+1}/{epochs} 시작 =====")
        
        epoch_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_roi_loss = 0.0
        epoch_temporal_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"에폭 {epoch+1}/{epochs}")
        
        # 새 에폭 시작 시 시간적 메모리 초기화
        model.reset_temporal_memory()
        
        # 시퀀스 정보 저장용 변수
        current_seq_id = None
        prev_frames = None
        
        for batch_idx, batch in enumerate(progress_bar):
            # 배치에서 시퀀스 ID 추출 (파일 경로에서)
            if "current_path" in batch:
                seq_ids = [os.path.basename(os.path.dirname(path)) for path in batch["current_path"]]
                
                # 시퀀스 ID가 변경되면 시간적 메모리 초기화
                if seq_ids[0] != current_seq_id:
                    model.reset_temporal_memory()
                    current_seq_id = seq_ids[0]
                    prev_frames = None
            
            # 시퀀스 기반 학습 스텝
            loss, loss_components, downsampled, mask, images = train_step(
                model, batch, optimizer, criterion, 
                object_detector=segmentation_model,
                device=device, 
                epoch=epoch,
                total_epochs=epochs,
                prev_frames=prev_frames
            )
            
            # 이전 프레임 정보 저장 (시퀀스 처리용)
            if use_sequences:
                prev_frames = images
            
            # 손실 누적
            epoch_loss += loss
            epoch_perceptual_loss += loss_components.get("perceptual", 0)
            epoch_roi_loss += loss_components.get("roi", 0)
            epoch_temporal_loss += loss_components.get("temporal", 0)
            
            # 진행 상황 업데이트
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "percept": f"{loss_components.get('perceptual', 0):.4f}",
                "roi": f"{loss_components.get('roi', 0):.4f}",
                "temp": f"{loss_components.get('temporal', 0):.4f}"
            })
            
            # 샘플 저장 (10 배치마다)
            if (batch_idx + 1) % 10 == 0:
                save_samples(
                    images, downsampled, mask, 
                    os.path.join(output_dir, "samples", f"epoch_{epoch+1}_batch_{batch_idx+1}.jpg"),
                    compare_masks=compare_masks
                )
        
        # 평균 손실 계산
        epoch_loss /= len(dataloader)
        epoch_perceptual_loss /= len(dataloader)
        epoch_roi_loss /= len(dataloader)
        epoch_temporal_loss /= len(dataloader)
        
        # 학습률 업데이트
        scheduler.step()
        
        # 통계 업데이트
        stats["epoch"].append(epoch + 1)
        stats["loss"].append(epoch_loss)
        stats["perceptual_loss"].append(epoch_perceptual_loss)
        stats["roi_loss"].append(epoch_roi_loss)
        stats["temporal_loss"].append(epoch_temporal_loss)
        stats["lr"].append(optimizer.param_groups[0]["lr"])
        
        # 에폭 결과 출력
        logger.info(f"에폭 {epoch+1} 결과: 손실={epoch_loss:.4f}, 지각적={epoch_perceptual_loss:.4f}, ROI={epoch_roi_loss:.4f}, 시간적={epoch_temporal_loss:.4f}")
        
        # 체크포인트 저장 (5 에폭마다)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            save_checkpoint(epoch + 1, epoch_loss)
            logger.info(f"체크포인트 저장 완료: 에폭 {epoch+1}")
            
            # 통계 저장
            with open(os.path.join(output_dir, "stats.json"), "w") as f:
                json.dump(stats, f, indent=4, cls=NumpyEncoder)
    
    # 최종 체크포인트 저장
    save_checkpoint(epochs, epoch_loss)
    logger.info("===== 학습 완료 =====")
    
    return model, stats

def main():
    """
    명령줄 인터페이스
    """
    parser = argparse.ArgumentParser(description="통합 ROI 다운샘플러 학습")
    parser.add_argument("--dataset_path", type=str, required=True, help="데이터셋 경로")
    parser.add_argument("--output_dir", type=str, default="output/roi_downsampler", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=15, help="에폭 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="가중치 감쇠")
    parser.add_argument("--scale_factor", type=int, default=2, help="다운샘플링 비율")
    parser.add_argument("--max_size", type=int, default=512, help="최대 이미지 크기")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="지각적 손실 가중치")
    parser.add_argument("--roi_weight", type=float, default=5.0, help="ROI 손실 가중치")
    parser.add_argument("--temporal_weight", type=float, default=2.0, help="시간적 일관성 손실 가중치")
    parser.add_argument("--mask_threshold", type=float, default=0.25, help="마스크 임계값")
    parser.add_argument("--compare_masks", action="store_true", help="마스크 비교 모드 활성화")
    parser.add_argument("--no_keep_aspect_ratio", action="store_true", help="종횡비 유지 비활성화")
    parser.add_argument("--clip_version", type=str, default="ViT-B/32", help="CLIP 모델 버전")
    parser.add_argument("--blip_version", type=str, default="Salesforce/blip2-opt-2.7b", help="BLIP 모델 버전")
    parser.add_argument("--yolo_version", type=str, default="yolov8n-seg.pt", help="YOLO 세그멘테이션 모델 버전")
    parser.add_argument("--r", type=float, default=0.5, help="배경 색상 R")
    parser.add_argument("--g", type=float, default=0.5, help="배경 색상 G")
    parser.add_argument("--b", type=float, default=0.5, help="배경 색상 B")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face 액세스 토큰")
    parser.add_argument("--no_sequences", action="store_true", help="시퀀스 모드 비활성화")
    parser.add_argument("--sequence_length", type=int, default=5, help="시퀀스 길이")
    parser.add_argument("--sequence_stride", type=int, default=2, help="시퀀스 스트라이드")
    parser.add_argument("--max_frames", type=int, default=100, help="시퀀스당 최대 프레임 수")
    
    args = parser.parse_args()
    
    # 배경 색상 설정
    bg_color = [args.r, args.g, args.b]
    
    # 모델 학습
    train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scale_factor=args.scale_factor,
        max_size=args.max_size,
        perceptual_weight=args.perceptual_weight,
        roi_weight=args.roi_weight,
        temporal_weight=args.temporal_weight,
        mask_threshold=args.mask_threshold,
        compare_masks=args.compare_masks,
        keep_aspect_ratio=not args.no_keep_aspect_ratio,
        clip_version=args.clip_version,
        blip_version=args.blip_version,
        yolo_version=args.yolo_version,
        bg_color=bg_color,
        hf_token=args.hf_token,
        use_sequences=not args.no_sequences,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        max_frames_per_sequence=args.max_frames
    )

if __name__ == "__main__":
    main() 