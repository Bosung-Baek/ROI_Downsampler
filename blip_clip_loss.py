import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import hashlib
from lpips import LPIPS

class BlipClipLoss(nn.Module):
    """
    BLIP2 + CLIP 기반 손실 함수
    
    BLIP2로 텍스트 추출 → CLIP 임베딩 → 텍스트-텍스트 유사도 비교를 통한 손실 계산
    """
    def __init__(
        self, 
        blip_model_name="Salesforce/blip2-opt-2.7b", 
        clip_model_name="ViT-B/32",
        device="cuda",
        hf_token=None
    ):
        """
        Args:
            blip_model_name: BLIP2 모델 이름
            clip_model_name: CLIP 모델 이름
            device: 연산에 사용할 장치
            hf_token: Hugging Face 액세스 토큰
        """
        super(BlipClipLoss, self).__init__()
        self.device = device
        
        # 토큰 설정
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print(f"Hugging Face 토큰 설정됨: {hf_token[:5]}...")
            
        # BLIP2 모델 로드
        print(f"BLIP2 모델 로드 중: {blip_model_name}")
        try:
            self.blip_processor = Blip2Processor.from_pretrained(blip_model_name)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                blip_model_name,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32
            ).to(device)
            print("BLIP2 모델 로드 완료")
        except Exception as e:
            print(f"BLIP2 모델 로드 실패: {e}")
            self.blip_processor = None
            self.blip_model = None
        
        # CLIP 모델 로드
        print(f"CLIP 모델 로드 중: {clip_model_name}")
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
            print("CLIP 모델 로드 완료")
        except Exception as e:
            print(f"CLIP 모델 로드 실패: {e}")
            self.clip_model = None
            self.clip_preprocess = None
    
    def tensor_to_pil(self, img_tensor):
        """이미지 텐서를 PIL 이미지로 변환"""
        img_tensor = img_tensor.detach().cpu().clamp(0, 1)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def generate_caption(self, image):
        """
        이미지에서 텍스트 캡션 생성
        
        Args:
            image: 이미지 텐서 [C, H, W] 또는 PIL 이미지
            
        Returns:
            생성된 텍스트 캡션 (문자열)
        """
        if self.blip_model is None:
            return f"img_caption_error_{hash(str(image))%10000}"
        
        # 텐서에서 PIL 이미지로 변환
        if isinstance(image, torch.Tensor):
            image = self.tensor_to_pil(image)
        
        # 간단한 프롬프트 사용
        prompt = "A photo of"
        
        try:
            # BLIP2 모델을 사용하여 캡션 생성
            inputs = self.blip_processor(image, text=prompt, return_tensors="pt").to(self.device)
            generated_ids = self.blip_model.generate(
                **inputs,
                max_new_tokens=50,
                min_length=5,
                num_beams=5,
                early_stopping=True,
                num_return_sequences=1
            )
            generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # 디버그 출력
            print(f"BLIP2 생성 캡션 (원본): {generated_text}")
            
            # 프롬프트 제거 확인
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                print(f"BLIP2 생성 캡션 (프롬프트 제거): {generated_text}")
            
            return generated_text
        except Exception as e:
            print(f"캡션 생성 오류: {e}")
            return f"img_caption_error_{hash(str(image))%10000}"
    
    def compute_text_embeddings(self, text):
        """
        텍스트 임베딩 계산
        
        Args:
            text: 텍스트 (문자열)
            
        Returns:
            텍스트 임베딩 벡터
        """
        if self.clip_model is None:
            return None
        
        # 텍스트 토큰화
        text_tokens = clip.tokenize([text]).to(self.device)
        
        # 임베딩 추출 및 정규화
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_caption_similarity(self, caption1, caption2):
        """
        두 캡션 간의 유사도 계산
        
        Args:
            caption1: 첫 번째 캡션 (문자열)
            caption2: 두 번째 캡션 (문자열)
            
        Returns:
            두 캡션 간의 코사인 유사도 (float)
        """
        if self.clip_model is None:
            return torch.tensor(0.5, device=self.device)
        
        try:
            # 다양한 프롬프트 제거
            prompts_to_remove = ["A photo of", "Describe the image:", "A picture of", "An image of", "This is a photo of"]
            
            for prompt in prompts_to_remove:
                if caption1.startswith(prompt):
                    caption1 = caption1[len(prompt):].strip()
                if caption2.startswith(prompt):
                    caption2 = caption2[len(prompt):].strip()
            
            # 디버깅 출력
            print(f"정제된 캡션1: {caption1}")
            print(f"정제된 캡션2: {caption2}")
            
            # 텍스트를 임베딩으로 변환
            text1_tokens = clip.tokenize([caption1]).to(self.device)
            text2_tokens = clip.tokenize([caption2]).to(self.device)
            
            text1_emb = self.clip_model.encode_text(text1_tokens)
            text2_emb = self.clip_model.encode_text(text2_tokens)
            
            # 정규화
            text1_emb = text1_emb / text1_emb.norm(dim=-1, keepdim=True)
            text2_emb = text2_emb / text2_emb.norm(dim=-1, keepdim=True)
            
            # 코사인 유사도 계산
            similarity = torch.matmul(text1_emb, text2_emb.T)[0][0]
            
            return similarity
        except Exception as e:
            print(f"캡션 유사도 계산 오류: {e}")
            return torch.tensor(0.5, device=self.device)
    
    def compute_clip_score(self, image, text):
        """
        이미지와 텍스트의 CLIP 점수 계산
        
        Args:
            image: 이미지 텐서 또는 PIL 이미지
            text: 텍스트 (문자열)
            
        Returns:
            CLIP 점수 (0~100)
        """
        if self.clip_model is None:
            return torch.tensor(0.0, device=self.device)
        
        # 이미지 전처리
        if isinstance(image, torch.Tensor):
            image = self.tensor_to_pil(image)
        clip_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # 텍스트 토큰화
        text_tokens = clip.tokenize([text]).to(self.device)
        
        # 임베딩 추출
        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_image)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # 정규화
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 유사도 계산
            similarity = (100.0 * (image_features @ text_features.T)).item()
        
        return similarity
    
    def forward(self, original_img, downsampled_img, object_mask=None):
        """
        손실 계산: 원본 이미지와 ROI 다운샘플링된 이미지(배경 회색)에서 생성한 캡션 간의 유사도 비교
        
        Args:
            original_img: 원본 이미지 텐서 [B, C, H, W]
            downsampled_img: ROI 다운샘플링된 이미지 텐서 [B, C, H', W'] (배경이 회색인 상태)
            object_mask: 객체 마스크 텐서 [B, 1, H', W'] (선택적)
            
        Returns:
            손실값 (텐서)
        """
        batch_size = original_img.shape[0]
        
        # 그래디언트 계산이 가능하도록 다운샘플링된 이미지 직접 사용
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 텍스트 캡션 및 유사도 저장용 리스트
        orig_captions = []
        down_captions = []
        similarities = []
        
        for i in range(batch_size):
            # 원본 이미지에서 캡션 생성
            with torch.no_grad():  # 캡션 생성은 그래디언트가 필요 없음
                orig_caption = self.generate_caption(original_img[i])
                orig_captions.append(orig_caption)
            
            # 다운샘플링된 이미지를 원본 크기로 업샘플링 (시각적 비교를 위해)
            if original_img.shape[2:] != downsampled_img.shape[2:]:
                downsampled_upscaled = F.interpolate(
                    downsampled_img[i].unsqueeze(0),
                    size=original_img.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                downsampled_upscaled = downsampled_img[i]
            
            # ROI 다운샘플링된 이미지(배경이 회색)에서 직접 캡션 생성
            with torch.no_grad():
                # 이미 배경이 회색인 상태의 이미지로 캡션 생성
                down_caption = self.generate_caption(downsampled_upscaled)
                down_captions.append(down_caption)
            
            # 텍스트 캡션 간 유사도 계산
            with torch.no_grad():
                caption_similarity = self.compute_caption_similarity(orig_caption, down_caption)
                similarities.append(caption_similarity)
                
                # 유사도가 낮을수록 손실이 높아야 함
                # 제곱을 사용하여 유사도 차이를 강화 (낮은 유사도에 더 큰 패널티)
                similarity_loss = 1.0 - (caption_similarity ** 2)
            
            # 디버깅 출력
            print(f"\n원본 캡션: {orig_caption}")
            print(f"ROI 다운샘플링 캡션: {down_caption}")
            print(f"캡션 유사도: {caption_similarity:.2f}, 손실: {similarity_loss:.4f}")
            
            # MSE 손실을 통해 그래디언트 계산이 가능한 손실 함수
            if i == 0:
                total_loss = F.mse_loss(downsampled_upscaled, original_img[i].detach()) * similarity_loss
            else:
                total_loss = total_loss + F.mse_loss(downsampled_upscaled, original_img[i].detach()) * similarity_loss
        
        # 평균 유사도 계산 (로깅용)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        print(f"배치 평균 캡션 유사도: {avg_similarity:.2f}")
        
        return total_loss / batch_size

class ROICompositeLoss(nn.Module):
    """
    ROI 기반 복합 손실 함수 (랜덤 마스킹 전략용)
    
    객체 탐지 성능 + 시맨틱 유사성 + 배경 영역 최적화를 위한 복합 손실
    """
    def __init__(
        self,
        od_weight=1.0,
        blip_clip_weight=1.0,
        roi_weight=2.0,
        masking_weight=1.5,
        hf_token=None,
        device="cuda"
    ):
        """
        Args:
            od_weight: 객체 탐지 손실 가중치
            blip_clip_weight: BLIP-CLIP 손실 가중치
            roi_weight: ROI 손실 가중치
            masking_weight: 마스킹 영역 최적화 손실 가중치
            hf_token: Hugging Face 토큰
            device: 연산 장치
        """
        super(ROICompositeLoss, self).__init__()
        
        # 손실 가중치
        self.od_weight = od_weight
        self.blip_clip_weight = blip_clip_weight
        self.roi_weight = roi_weight
        self.masking_weight = masking_weight
        
        # 장치 설정
        self.device = device
        
        # BLIP-CLIP 손실 모듈 (의미적 유사성)
        self.blip_clip_module = BlipClipLoss(
            blip_model_name="Salesforce/blip2-opt-2.7b", 
            clip_model_name="ViT-B/32",
            device=device,
            hf_token=hf_token
        )
        
        # 객체 탐지기 (없을 경우 None)
        self.detector = None
        
        # LPIPS 손실 초기화
        try:
            self.lpips = LPIPS(net='alex').to(device)
            print("LPIPS 손실 초기화 완료")
        except Exception as e:
            print(f"LPIPS 초기화 오류: {e}. 기본 MSE 손실을 대신 사용합니다.")
            self.lpips = None
    
    def set_object_detector(self, detector):
        """
        객체 탐지기 설정
        
        Args:
            detector: YOLO 객체 탐지기
        """
        self.detector = detector
    
    def object_detection_loss(self, original_img, downsampled_img, object_mask=None):
        """
        객체 탐지 손실: 원본 이미지와 ROI 다운샘플링된 이미지에서 객체 탐지 결과 비교
        
        Args:
            original_img: 원본 이미지 [B, C, H, W]
            downsampled_img: ROI 다운샘플링된 이미지 텐서 [B, C, H', W'] (배경이 회색인 상태)
            object_mask: 객체 마스크 [B, 1, H, W]
            
        Returns:
            객체 탐지 손실 (텐서)
        """
        # 객체 탐지기가 없는 경우 0 반환
        if self.detector is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 배치 크기
        batch_size = original_img.shape[0]
        
        # 그래디언트 계산이 가능한 손실 초기화
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i in range(batch_size):
            # 원본 이미지를 CPU로 변환 (numpy 배열)
            orig_np = (original_img[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
            
            # 다운샘플링된 이미지를 원본 크기로 업샘플링
            if original_img.shape[2:] != downsampled_img.shape[2:]:
                down_upsampled = F.interpolate(
                    downsampled_img[i].unsqueeze(0),
                    size=original_img.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                down_upsampled = downsampled_img[i]
            
            # ROI 다운샘플링된 이미지(배경이 회색)를 직접 사용
            # numpy 배열로 변환
            down_np = (down_upsampled.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
            
            # 객체 탐지
            with torch.no_grad():
                results_orig = self.detector.predict(orig_np, verbose=False)[0]
                results_down = self.detector.predict(down_np, verbose=False)[0]
            
            # 바운딩 박스 IoU 계산 (객체가 있는 경우)
            if len(results_orig.boxes) > 0 and len(results_down.boxes) > 0:
                boxes_orig = results_orig.boxes.xyxy.cpu().numpy()
                boxes_down = results_down.boxes.xyxy.cpu().numpy()
                
                # 각 객체별 최대 IoU 계산
                max_ious = []
                for box_orig in boxes_orig:
                    ious = []
                    for box_down in boxes_down:
                        # IoU 계산
                        x1 = max(box_orig[0], box_down[0])
                        y1 = max(box_orig[1], box_down[1])
                        x2 = min(box_orig[2], box_down[2])
                        y2 = min(box_orig[3], box_down[3])
                        
                        if x2 > x1 and y2 > y1:
                            intersection = (x2 - x1) * (y2 - y1)
                            area_orig = (box_orig[2] - box_orig[0]) * (box_orig[3] - box_orig[1])
                            area_down = (box_down[2] - box_down[0]) * (box_down[3] - box_down[1])
                            union = area_orig + area_down - intersection
                            iou = intersection / union
                        else:
                            iou = 0.0
                        
                        ious.append(iou)
                    
                    # 최대 IoU 추가
                    max_ious.append(max(ious) if ious else 0.0)
                
                # IoU 손실: 1 - 평균 IoU
                avg_iou = np.mean(max_ious)
                iou_loss = 1.0 - avg_iou
                
                # MSE 손실을 통해 그래디언트 계산이 가능한 손실 추가
                # IoU가 높을수록 객체 영역 손실 감소
                iou_loss_tensor = F.mse_loss(
                    down_upsampled, 
                    original_img[i].detach()
                ) * iou_loss
                
                total_loss = total_loss + iou_loss_tensor
            
            # 객체 신뢰도 점수 차이
            conf_orig = results_orig.boxes.conf.mean().item() if len(results_orig.boxes) > 0 else 0
            conf_down = results_down.boxes.conf.mean().item() if len(results_down.boxes) > 0 else 0
            conf_diff = abs(conf_orig - conf_down)
            
            # MSE 손실을 통해 그래디언트 계산이 가능한 손실 추가
            conf_loss_tensor = F.mse_loss(down_upsampled, original_img[i].detach()) * conf_diff
            total_loss = total_loss + conf_loss_tensor
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def blip_clip_loss(self, original_img, downsampled_img, object_mask=None):
        """
        BLIP-CLIP 손실: 이미지 캡션의 의미적 유사성 비교
        
        Args:
            original_img: 원본 이미지 [B, C, H, W]
            downsampled_img: ROI 다운샘플링된 이미지 텐서 [B, C, H', W'] (배경이 회색인 상태)
            object_mask: 객체 마스크 [B, 1, H, W]
            
        Returns:
            BLIP-CLIP 손실 (텐서)
        """
        # 객체 마스크에 관계없이 BLIP-CLIP 모듈로 손실 계산
        loss = self.blip_clip_module(original_img, downsampled_img, object_mask)
        return loss
    
    def adaptive_masking_loss(self, mask, randomness_factor=0.3, epoch=0):
        """
        적응형 마스킹 손실: 마스킹 영역의 최적 비율과 랜덤성 촉진
        
        Args:
            mask: 마스크 텐서 [B, 1, H, W]
            randomness_factor: 마스킹의 랜덤성을 촉진하기 위한 가중치
            epoch: 현재 학습 에폭 (점진적 학습을 위해 사용)
            
        Returns:
            마스킹 손실 (텐서)
        """
        # 마스크가 없는 경우 0 반환
        if mask is None:
            print("경고: adaptive_masking_loss에 마스크가 None입니다.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # NaN/Inf 값 체크
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print("경고: adaptive_masking_loss에 마스크에 NaN/Inf 값이 있습니다.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            # 마스크 비율 계산 (객체 영역 비율)
            mask_ratio = torch.mean(mask)
            
            # 에폭에 따른 목표 마스크 비율 계산 (점진적으로 감소)
            # 초기: 객체 영역 30-50% 허용
            # 후기: 객체 영역 10-30% 허용 (더 많은 압축)
            target_min = max(0.1, 0.3 - 0.02 * epoch)  # 에폭에 따라 감소
            target_max = max(0.3, 0.5 - 0.02 * epoch)  # 에폭에 따라 감소
            
            # 디버그 출력 (가끔)
            if torch.rand(1).item() < 0.01:  # 1% 확률로 출력
                print(f"에폭 {epoch}: 목표 마스크 비율 {target_min:.2f}-{target_max:.2f}, 현재={mask_ratio.item():.2f}")
            
            # 마스크 비율 조절 손실 (적정 비율로 유도)
            if mask_ratio < target_min:
                # 마스크 비율이 너무 낮으면 증가 유도
                ratio_loss = (target_min - mask_ratio) ** 2
            elif mask_ratio > target_max:
                # 마스크 비율이 너무 높으면 감소 유도
                ratio_loss = (mask_ratio - target_max) ** 2
            else:
                # 적정 범위 내에 있으면 0
                ratio_loss = torch.tensor(0.0, device=self.device)
            
            # 마스크의 랜덤성 촉진 (배경 영역만)
            # 0.5 임계값으로 객체/배경 구분
            binary_mask = (mask > 0.5).float()
            background_mask = 1.0 - binary_mask
            
            # 분모가 0이 되지 않도록 보호
            sum_bg = torch.sum(background_mask)
            if sum_bg < 1e-6:  # 배경 영역이 거의 없으면
                print("경고: 배경 영역이 거의 없습니다.")
                return ratio_loss  # 비율 손실만 반환
            
            # 엔트로피 계산 (높은 엔트로피 = 높은 랜덤성)
            # 배경 영역의 마스크 값이 다양할수록 엔트로피 증가
            eps = 1e-8
            mask_clipped = torch.clamp(mask, eps, 1.0 - eps)
            entropy = -(mask_clipped * torch.log(mask_clipped) + 
                        (1.0 - mask_clipped) * torch.log(1.0 - mask_clipped))
            
            # 배경 영역의 엔트로피만 고려
            bg_entropy = torch.sum(entropy * background_mask) / (sum_bg + eps)
            
            # 에폭에 따른 랜덤성 가중치 조정 (초기: 높은 랜덤성, 후기: 낮은 랜덤성)
            epoch_randomness = randomness_factor * max(1.0 - epoch / 20.0, 0.2)
            
            # 랜덤성 손실 (엔트로피가 높을수록 손실 감소)
            randomness_loss = -bg_entropy * epoch_randomness
            
            # 전체 마스킹 손실
            total_loss = ratio_loss + randomness_loss
            
            # NaN 검사
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("경고: 마스킹 손실이 NaN 또는 Inf입니다.")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            return total_loss
            
        except Exception as e:
            print(f"adaptive_masking_loss 오류: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def forward(self, original_img, downsampled_img, object_mask=None, all_masks=None, epoch=0):
        """
        ROI 다운샘플링 이미지에 대한 손실 계산
        
        Args:
            original_img: 원본 이미지 [B, C, H, W]
            downsampled_img: ROI 다운샘플링된 이미지 텐서 [B, C, H', W'] (배경이 회색인 상태)
            object_mask: 객체 마스크 [B, 1, H, W]
            all_masks: 다양한 마스크 정보 (세그멘테이션, 생성된 마스크 등) - ROI 다운샘플링에서는 사용하지 않음
            epoch: 현재 학습 에폭 (점진적 학습을 위해 사용)
            
        Returns:
            전체 손실, 손실 구성요소 딕셔너리
        """
        # 손실 계산 전에 NaN/Inf 체크
        if torch.isnan(original_img).any() or torch.isinf(original_img).any() or \
           torch.isnan(downsampled_img).any() or torch.isinf(downsampled_img).any():
            print("NaN/Inf 입력 감지됨")
            return torch.tensor(0.0, device=self.device), {'total': 0.0, 'object_detection': 0.0, 'blip_clip': 0.0}
        
        # 1. 객체 탐지 손실
        od_loss = self.object_detection_loss(original_img, downsampled_img, object_mask)
        
        # 2. BLIP-CLIP 손실
        blip_clip_loss = self.blip_clip_loss(original_img, downsampled_img, object_mask)
        
        # ROI 다운샘플링에서는 마스킹 손실을 사용하지 않음 (배경이 이미 회색으로 처리되어 있으므로)
        
        # 손실 가중치 설정
        od_weight = self.od_weight * 1.5  # 객체 탐지 손실 가중치 증가
        blip_clip_weight = self.blip_clip_weight
        
        # 가중치 적용된 손실
        weighted_od_loss = od_weight * od_loss
        weighted_blip_clip_loss = blip_clip_weight * blip_clip_loss
        
        # 전체 손실
        total_loss = weighted_od_loss + weighted_blip_clip_loss
        
        # 학습 상태 출력 (10% 확률)
        if torch.rand(1).item() < 0.1:
            print(f"에폭 {epoch} 손실 - 총: {total_loss.item():.4f}, OD: {od_loss.item():.4f}({od_weight:.2f}), "
                  f"BLIP-CLIP: {blip_clip_loss.item():.4f}({blip_clip_weight:.2f})")
        
        # 학습 진행 모니터링을 위한 손실 요소 딕셔너리
        loss_components = {
            'total': total_loss.item(),
            'object_detection': od_loss.item(),
            'blip_clip': blip_clip_loss.item(),
            'weighted_od': weighted_od_loss.item(),
            'weighted_blip_clip': weighted_blip_clip_loss.item()
        }
        
        return total_loss, loss_components 

# 새로운 통합 손실 함수
class IntegratedLoss(nn.Module):
    """통합 ROI 다운샘플러 손실 함수"""
    def __init__(self, perceptual_weight=1.0, roi_weight=1.0, temporal_weight=1.0, clip_model=None, blip_processor=None, blip_model=None, device="cuda"):
        super(IntegratedLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.roi_weight = roi_weight
        self.temporal_weight = temporal_weight
        self.device = device
        
        # CLIP 및 BLIP 모델 저장
        self.clip_model = clip_model
        self.blip_processor = blip_processor
        self.blip_model = blip_model
        
        # LPIPS 손실 초기화
        try:
            self.lpips = LPIPS(net='alex').to(device)
            print("LPIPS 손실 초기화 완료")
        except Exception as e:
            print(f"LPIPS 초기화 오류: {e}. 기본 MSE 손실을 대신 사용합니다.")
            self.lpips = None
        
        # 이전 프레임 기록 (시간적 일관성)
        self.prev_frame = None
        self.prev_downsampled = None
    
    def forward(self, original_img, downsampled_img, object_mask=None, mask=None, epoch=0):
        """
        손실 계산
        
        Args:
            original_img: 원본 이미지 텐서 [B, C, H, W]
            downsampled_img: 다운샘플링된 이미지 텐서 [B, C, H', W']
            object_mask: 객체 마스크 [B, 1, H, W]
            mask: 생성된 마스크 [B, 1, H, W]
            epoch: 현재 학습 에폭
            
        Returns:
            total_loss, loss_components
        """
        batch_size = original_img.shape[0]
        loss_components = {}
        
        # 원본 크기로 리사이징하여 손실 계산
        if downsampled_img.shape[2:] != original_img.shape[2:]:
            upsampled = F.interpolate(
                downsampled_img, 
                size=(original_img.shape[2], original_img.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            upsampled = downsampled_img
        
        # 1. 기본 재구성 손실 (L1 + SSIM)
        # L1 손실
        l1_loss = F.l1_loss(upsampled, original_img)
        
        # SSIM 손실
        ssim_loss = 1.0 - self.calculate_ssim(upsampled, original_img)
        
        # 재구성 손실 = L1 + SSIM
        recon_loss = 0.5 * l1_loss + 0.5 * ssim_loss
        loss_components["recon"] = recon_loss.item()
        
        # 2. ROI 기반 손실 (객체 영역만 강조)
        if object_mask is not None:
            # 객체 영역만 손실 계산 (object_mask의 0 영역 = 객체)
            roi_mse = F.mse_loss(upsampled, original_img, reduction='none')
            
            # 마스크 크기 조정
            if object_mask.shape[2:] != original_img.shape[2:]:
                object_mask_resized = F.interpolate(
                    object_mask, 
                    size=(original_img.shape[2], original_img.shape[3]), 
                    mode='nearest'
                )
            else:
                object_mask_resized = object_mask
            
            # 객체 마스크 반전 (0=객체, 1=배경)
            roi_mask = 1.0 - object_mask_resized
            
            # 객체 영역에 대한 MSE 손실
            roi_loss = (roi_mse * roi_mask).sum() / (roi_mask.sum() * 3 + 1e-8)
            loss_components["roi"] = roi_loss.item()
        else:
            roi_loss = torch.tensor(0.0, device=self.device)
            loss_components["roi"] = 0.0
        
        # 3. 지각적 손실 (LPIPS)
        if self.lpips is not None:
            try:
                perc_loss = self.lpips(original_img, upsampled).mean()
                loss_components["perceptual"] = perc_loss.item()
            except Exception as e:
                print(f"LPIPS 계산 오류: {e}. 건너뜁니다.")
                perc_loss = torch.tensor(0.0, device=self.device)
                loss_components["perceptual"] = 0.0
        else:
            perc_loss = torch.tensor(0.0, device=self.device)
            loss_components["perceptual"] = 0.0
        
        # 4. 마스크 손실 (평활도 및 면적 제약)
        if mask is not None:
            # 마스크 그라디언트 계산 (x, y 방향)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float().view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).float().view(1, 1, 3, 3)
            
            # 패딩 추가
            padded_mask = F.pad(mask, (1, 1, 1, 1), mode='replicate')
            
            # 그라디언트 계산 (컨볼루션)
            grad_x = F.conv2d(padded_mask, sobel_x)
            grad_y = F.conv2d(padded_mask, sobel_y)
            
            # 그라디언트 크기
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
            
            # 마스크 평활도 손실
            smoothness_loss = grad_mag.mean()
            
            # 마스크 면적 제약 (목표 비율: 에폭에 따라 조정)
            target_ratio = max(0.5 - 0.02 * epoch, 0.2)  # 에폭이 증가할수록 더 적은 배경 영역
            area_loss = torch.abs(mask.mean() - target_ratio)
            
            # 최종 마스크 손실
            mask_loss = 0.7 * smoothness_loss + 0.3 * area_loss
            loss_components["mask"] = mask_loss.item()
        else:
            mask_loss = torch.tensor(0.0, device=self.device)
            loss_components["mask"] = 0.0
        
        # 5. 시간적 일관성 손실
        if self.prev_frame is not None and self.prev_downsampled is not None:
            # 현재 프레임과 이전 프레임 간의 변화
            frame_diff = F.l1_loss(original_img, self.prev_frame, reduction='none')
            
            # 다운샘플링된 이미지 간의 변화 (동일한 크기로 조정)
            if self.prev_downsampled.shape[2:] != downsampled_img.shape[2:]:
                prev_ds_resized = F.interpolate(
                    self.prev_downsampled, 
                    size=downsampled_img.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                prev_ds_resized = self.prev_downsampled
            
            ds_diff = F.l1_loss(downsampled_img, prev_ds_resized, reduction='none')
            
            # 시간적 일관성 손실: 원본 변화량과 다운샘플 변화량의 불일치 정도
            temp_loss = torch.abs(frame_diff.mean() - ds_diff.mean())
            loss_components["temporal"] = temp_loss.item()
        else:
            temp_loss = torch.tensor(0.0, device=self.device)
            loss_components["temporal"] = 0.0
        
        # 이전 프레임과 다운샘플링 결과 저장
        self.prev_frame = original_img.detach()
        self.prev_downsampled = downsampled_img.detach()
        
        # 총 손실 계산
        total_loss = recon_loss + self.roi_weight * roi_loss + self.perceptual_weight * perc_loss + mask_loss + self.temporal_weight * temp_loss
        
        return total_loss, loss_components
    
    def calculate_ssim(self, img1, img2, window_size=11, size_average=True):
        """SSIM 계산 함수"""
        # 창 함수 생성
        window = self.create_window(window_size, img1.shape[1]).to(img1.device)
        
        # 값 범위 확인
        if img1.min() < 0 or img1.max() > 1 or img2.min() < 0 or img2.max() > 1:
            img1 = torch.clamp(img1, 0, 1)
            img2 = torch.clamp(img2, 0, 1)
        
        # 채널 차원을 배치 차원으로 변환
        batch_size = img1.shape[0]
        channel = img1.shape[1]
        
        # 각 채널을 배치로 취급
        img1 = img1.view(batch_size * channel, 1, img1.shape[2], img1.shape[3])
        img2 = img2.view(batch_size * channel, 1, img2.shape[2], img2.shape[3])
        
        # 평균 계산
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=batch_size*channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=batch_size*channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 분산 계산
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=batch_size*channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=batch_size*channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=batch_size*channel) - mu1_mu2
        
        # SSIM 상수
        C1 = 0.01**2
        C2 = 0.03**2
        
        # SSIM 공식
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def create_window(self, window_size, channel):
        """가우시안 창 함수 생성"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window 