import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DownsamplerBlock(nn.Module):
    """
    다운샘플링 블록
    
    입력 이미지를 2배 또는 4배 다운샘플링하는 기본 블록
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownsamplerBlock, self).__init__()
        
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x  # Skip connection을 위한 원본 입력 저장
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class ResBlock(nn.Module):
    """
    잔차 블록
    
    특징 정보를 유지하면서 모델 깊이를 증가시키는 블록
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 입력과 출력 채널 수가 다른 경우 1x1 컨볼루션으로 조정
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 입력과 출력 채널 수가 다른 경우 조정
        if self.skip is not None:
            identity = self.skip(identity)
            
        out += identity
        out = F.relu(out)
        
        return out

class RandomMaskGenerator(nn.Module):
    """
    학습 가능한 마스크 생성기
    
    객체 인식 성능과 압축률 사이의 최적점을 학습하는 어텐션 기반 네트워크
    """
    def __init__(self, in_channels=3, features=64, randomness=0.5):
        super(RandomMaskGenerator, self).__init__()
        self.randomness = randomness
        self.transition_zone = 3  # 전이 영역 크기
        self.training_epoch = 0  # 현재 학습 에폭 추적
        
        # 인코더 (특징 추출)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(),
        )
        
        # 어텐션 모듈 (중요 영역 식별)
        self.attention = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(),
        )
        
        # 디코더 (원본 해상도로 복원)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features//2, features//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(),
            nn.ConvTranspose2d(features//2, features//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU(),
            nn.Conv2d(features//4, 1, kernel_size=3, stride=1, padding=1),
        )
        
        # 객체 마스크 통합 네트워크
        self.object_integration = nn.Sequential(
            nn.Conv2d(1, features//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU(),
            nn.Conv2d(features//4, features//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU(),
            nn.Conv2d(features//4, 1, kernel_size=1, stride=1, padding=0),
        )
        
        # 노이즈 스케일 (학습 가능 파라미터)
        self.noise_scale = nn.Parameter(torch.tensor([0.5]))
        
        # 최소 마스크 비율 (에폭에 따라 감소)
        self.min_mask_ratio = 0.4
    
    def set_epoch(self, epoch):
        """현재 에폭 설정 (점진적 학습용)"""
        self.training_epoch = epoch
    
    def get_target_mask_ratio(self):
        """에폭에 따른 목표 마스크 비율 계산"""
        # 에폭이 진행될수록 마스크 비율을 줄여감 (더 많은 압축)
        return max(0.4 - 0.05 * self.training_epoch, 0.1)
    
    def forward(self, x, object_mask=None, training=True):
        batch_size = x.shape[0]
        device = x.device
        
        # 중요 영역 특징 추출 (인코더)
        features = self.encoder(x)
        
        # 어텐션 맵 생성
        attention_features = self.attention(features)
        
        # 마스크 생성 (디코더)
        mask_logits = self.decoder(attention_features)
        
        # 객체 마스크 통합 (있는 경우)
        if object_mask is not None:
            # 객체 마스크 특징 추출
            object_features = self.object_integration(object_mask)
            
            # 가중치 계산 (학습 진행에 따라 조정)
            # 초기에는 객체 마스크에 크게 의존, 나중에는 학습된 어텐션에 더 의존
            epoch_weight = min(self.training_epoch / 10.0, 1.0)
            
            # 두 마스크 결합 (학습 초기: 객체 마스크 중심, 후기: 학습된 어텐션 중심)
            combined_mask = (1.0 - epoch_weight) * object_features + epoch_weight * mask_logits
        else:
            combined_mask = mask_logits
        
        # 학습 중이고 랜덤성 활성화된 경우 노이즈 추가
        if training and self.randomness > 0:
            # 에폭에 따라 노이즈 감소 (초기: 많은 탐색, 후기: 안정화)
            epoch_noise_scale = self.noise_scale * max(1.0 - (self.training_epoch / 20.0), 0.1)
            
            # 랜덤 노이즈 생성
            noise = torch.rand_like(combined_mask) * 2 - 1  # -1~1 범위
            noise = noise * epoch_noise_scale
            
            # 객체 영역에는 노이즈 감소
            if object_mask is not None:
                noise = noise * (1.0 - object_mask * 0.8)
            
            # 노이즈 적용
            combined_mask = combined_mask + noise
        
        # 마스크 활성화
        mask = torch.sigmoid(combined_mask)
        
        # 에폭별 목표 마스크 비율에 맞도록 임계값 조정
        if training:
            target_ratio = self.get_target_mask_ratio()
            current_ratio = torch.mean(mask)
            
            # 디버그 출력
            if batch_size > 0 and torch.rand(1).item() < 0.01:  # 1% 확률로 출력
                print(f"에폭 {self.training_epoch}: 목표 마스크 비율={target_ratio:.3f}, 현재={current_ratio.item():.3f}")
        
        return mask
    
    def __call__(self, binary_mask):
        """
        이전 버전과의 호환성을 위한 인터페이스
        학습 과정에서는 forward 메서드가 사용됨
        """
        import numpy as np
        from scipy import ndimage
        from skimage import morphology
        
        # 입력 텐서의 장치 기억
        device = binary_mask.device if torch.is_tensor(binary_mask) else None
        
        try:
            # 학습 초기 (0-3 에폭): 객체 마스크 기반 + 랜덤성 
            # 학습 중기 (4-10 에폭): 부분적으로 학습된 어텐션
            # 학습 후기 (11+ 에폭): 완전히 학습된 어텐션 사용
            
            # 현재 에폭 확인하여 접근 방식 결정
            if self.training_epoch < 4:
                # 마스크 형태 변환 (텐서→넘파이)
                if torch.is_tensor(binary_mask):
                    # 배치 차원 처리
                    if binary_mask.dim() == 4:
                        binary_mask = binary_mask[0].squeeze(0).cpu().numpy()
                    elif binary_mask.dim() == 3:
                        binary_mask = binary_mask.squeeze(0).cpu().numpy()
                    else:
                        binary_mask = binary_mask.cpu().numpy()
                        
                    # 1D 배열 처리
                    if binary_mask.ndim == 1:
                        binary_mask = np.expand_dims(binary_mask, axis=0)
                
                # 이진화
                binary_mask = (binary_mask > 0.5).astype(np.float32)
                
                # 전이 영역 생성
                try:
                    dilated = morphology.binary_dilation(binary_mask, morphology.disk(self.transition_zone))
                    eroded = morphology.binary_erosion(binary_mask, morphology.disk(self.transition_zone))
                    transition_mask = dilated & ~eroded
                except Exception as e:
                    print(f"전이 영역 생성 오류: {e}")
                    transition_mask = np.zeros_like(binary_mask, dtype=bool)
                
                # 랜덤 마스크 초기화
                final_mask = np.copy(binary_mask).astype(np.float32)
                
                # 객체 영역 처리
                if np.sum(binary_mask > 0) > 0:
                    # 에폭에 따라 보존 수준 조정 (점점 줄어듦)
                    obj_preserve = max(0.85 - self.training_epoch * 0.05, 0.6)
                    final_mask[binary_mask > 0] = np.random.uniform(obj_preserve, 1.0, size=np.sum(binary_mask > 0))
                
                # 배경 영역 처리
                if np.sum(binary_mask == 0) > 0:
                    # 에폭에 따라 배경 보존 감소 (점점 더 많이 압축)
                    bg_max = min(0.15 + self.training_epoch * 0.02, 0.3)
                    final_mask[binary_mask == 0] = np.random.uniform(0.0, bg_max, size=np.sum(binary_mask == 0))
                
                # 전이 영역 처리
                if np.sum(transition_mask) > 0:
                    final_mask[transition_mask] = np.random.uniform(0.3, 0.7, size=np.sum(transition_mask))
                
                # 텐서로 변환 및 차원 조정
                result = torch.tensor(final_mask, dtype=torch.float32)
                while result.dim() < 4:
                    result = result.unsqueeze(0)
            else:
                # 학습 중기/후기에는 학습된 어텐션 활용
                # binary_mask를 입력으로 사용하여 forward 호출
                if torch.is_tensor(binary_mask):
                    if binary_mask.dim() == 4:
                        # 이미 [B, C, H, W] 형태
                        obj_input = binary_mask
                    elif binary_mask.dim() == 3:
                        # [C, H, W] -> [1, C, H, W]
                        obj_input = binary_mask.unsqueeze(0)
                    elif binary_mask.dim() == 2:
                        # [H, W] -> [1, 1, H, W]
                        obj_input = binary_mask.unsqueeze(0).unsqueeze(0)
                    else:
                        # [W] -> [1, 1, 1, W]
                        obj_input = binary_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        h = 1  # 임의의 높이
                        obj_input = obj_input.expand(1, 1, h, -1)
                    
                    # 장치 이동
                    obj_input = obj_input.to(device) if device is not None else obj_input
                    
                    # 더미 이미지 생성 (실제로는 사용되지 않음)
                    dummy_img = torch.zeros_like(obj_input)
                    
                    # forward 메서드 호출하여 학습된 어텐션 기반 마스크 생성
                    result = self.forward(dummy_img, obj_input, training=False)
                else:
                    # NumPy 배열인 경우 텐서로 변환
                    obj_input = torch.tensor(binary_mask, dtype=torch.float32)
                    while obj_input.dim() < 4:
                        obj_input = obj_input.unsqueeze(0)
                    
                    dummy_img = torch.zeros_like(obj_input)
                    result = self.forward(dummy_img, obj_input, training=False)
            
            # 장치 확인 및 반환
            return result.to(device) if device is not None else result
        
        except Exception as e:
            print(f"마스크 생성 오류: {e}")
            # 오류 시 기본 마스크 반환
            if torch.is_tensor(binary_mask):
                if binary_mask.dim() == 4:
                    h, w = binary_mask.shape[2], binary_mask.shape[3]
                elif binary_mask.dim() == 3:
                    h, w = binary_mask.shape[1], binary_mask.shape[2]
                elif binary_mask.dim() == 2:
                    h, w = binary_mask.shape
                else:
                    return torch.zeros((1, 1, 1, binary_mask.shape[0]), dtype=torch.float32, device=device)
                
                return torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
            else:
                if len(binary_mask.shape) >= 2:
                    h, w = binary_mask.shape[-2], binary_mask.shape[-1]
                    return torch.zeros((1, 1, h, w), dtype=torch.float32)
                else:
                    return torch.zeros((1, 1, 1, len(binary_mask)), dtype=torch.float32)

class TemporalConsistencyModule(nn.Module):
    """시간적 일관성 모듈"""
    def __init__(self, channels=64, hidden_dim=128):
        super(TemporalConsistencyModule, self).__init__()
        
        # 현재 프레임과 이전 프레임 특징을 결합하는 컨볼루션
        self.fusion_conv = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1)
        self.fusion_bn = nn.BatchNorm2d(channels)
        
        # 시간적 가중치 예측 네트워크
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 초기 가중치 설정
        self.default_weights = nn.Parameter(torch.tensor([0.7, 0.3]), requires_grad=False)
    
    def forward(self, current_features, prev_features=None):
        # 이전 프레임 특징이 없는 경우
        if prev_features is None:
            return current_features, self.default_weights
        
        # 현재 및 이전 프레임 특징 결합
        fused_features = torch.cat([current_features, prev_features], dim=1)
        fused_features = F.relu(self.fusion_bn(self.fusion_conv(fused_features)))
        
        # 시간적 가중치 예측
        weights = self.weight_predictor(fused_features)
        
        # 차원 변환 (batch_size, 2, 1, 1) -> (batch_size, 2)
        weights = weights.mean([2, 3])
        
        # 현재 및 이전 프레임의 가중치 적용
        temporally_consistent_features = weights[:, 0:1].unsqueeze(2).unsqueeze(3) * current_features + \
                                         weights[:, 1:2].unsqueeze(2).unsqueeze(3) * prev_features
        
        return temporally_consistent_features, weights

class SequenceAnalysisModule(nn.Module):
    """시퀀스 특성 분석 모듈"""
    def __init__(self, in_channels=3, features=64):
        super(SequenceAnalysisModule, self).__init__()
        
        # 입력 이미지 특징 추출
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(features, features*2),
            ResBlock(features*2, features*2),
        )
        
        # 시퀀스 분석 (전역 특징 기반)
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(features*2, features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features//2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
        # 객체 복잡도 분석
        self.object_complexity = nn.Sequential(
            nn.Linear(features//2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 움직임 분석
        self.motion_analysis = nn.Sequential(
            nn.Linear(features//2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
        
        # 장면 분석
        self.scene_analysis = nn.Sequential(
            nn.Linear(features//2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )
        
        # 최종 특성 결합 및 파라미터 예측
        self.parameter_predictor = nn.Sequential(
            nn.Linear(1 + 2 + 3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 9)  # 9개의 파라미터 예측
        )
    
    def forward(self, x, prev_frames=None):
        batch_size = x.shape[0]
        
        # 현재 프레임 특징 추출
        image_features = self.image_encoder(x)
        
        # 전역 특징 추출
        global_feat = self.global_features(image_features)
        global_feat_flat = global_feat.view(batch_size, -1)
        
        # 특성 분석
        obj_complexity = self.object_complexity(global_feat_flat)
        motion_feat = self.motion_analysis(global_feat_flat)
        scene_feat = self.scene_analysis(global_feat_flat)
        
        # 특성 결합
        combined_features = torch.cat([obj_complexity, motion_feat, scene_feat], dim=1)
        
        # 파라미터 예측
        params_raw = self.parameter_predictor(combined_features)
        
        # 파라미터 범위 조정
        params = {
            'conf_threshold': torch.sigmoid(params_raw[:, 0]) * 0.1 + 0.01,  # 0.01-0.11
            'dilate_kernel_size': torch.clamp(torch.round(params_raw[:, 1] * 4 + 7), 3, 15),  # 3-15, 홀수 값
            'mask_threshold': torch.sigmoid(params_raw[:, 2]) * 0.25 + 0.1,  # 0.1-0.35
            'iou_threshold': torch.sigmoid(params_raw[:, 3]) * 0.2 + 0.15,  # 0.15-0.35
            'mask_decay': torch.sigmoid(params_raw[:, 4]) * 0.2 + 0.8,  # 0.8-1.0
            'weight_current': torch.sigmoid(params_raw[:, 5]) * 0.5 + 0.3,  # 0.3-0.8
            'weight_previous': None,  # 계산됨
            'blur_kernel_size': torch.clamp(torch.round(params_raw[:, 7] * 4 + 5), 3, 13),  # 3-13, 홀수 값
            'margin_factor': torch.sigmoid(params_raw[:, 8]) * 0.07 + 0.01  # 0.01-0.08
        }
        
        # weight_previous 계산
        params['weight_previous'] = 1.0 - params['weight_current']
        
        return params, image_features

class AdaptiveMaskGenerator(nn.Module):
    """적응형 마스크 생성기"""
    def __init__(self, in_channels=3, features=64):
        super(AdaptiveMaskGenerator, self).__init__()
        
        # 인코더 (특징 추출)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True),
        )
        
        # 어텐션 모듈 (중요 영역 식별)
        self.attention = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(inplace=True),
        )
        
        # 디코더 (원본 해상도로 복원)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features//2, features//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features//2, features//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//4, 1, kernel_size=3, stride=1, padding=1),
        )
        
        # 객체 마스크 통합 네트워크
        self.object_integration = nn.Sequential(
            nn.Conv2d(1, features//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//4, features//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//4, 1, kernel_size=1, stride=1, padding=0),
        )
        
        # 시간적 일관성 모듈
        self.temporal_consistency = TemporalConsistencyModule(features//4)
    
    def forward(self, x, object_mask=None, prev_mask=None, prev_features=None):
        batch_size = x.shape[0]
        device = x.device
        
        # 중요 영역 특징 추출 (인코더)
        features = self.encoder(x)
        
        # 어텐션 맵 생성
        attention_features = self.attention(features)
        
        # 마스크 생성 (디코더)
        mask_features = self.decoder(attention_features)
        
        # 객체 마스크 통합 (있는 경우)
        if object_mask is not None:
            # 객체 마스크 특징 추출
            object_features = self.object_integration(object_mask)
            
            # 객체 특징과 마스크 특징 결합
            combined_features = mask_features + object_features
        else:
            combined_features = mask_features
        
        # 시간적 일관성 적용 (이전 마스크가 있는 경우)
        if prev_features is not None:
            tmp_features, weights = self.temporal_consistency(combined_features, prev_features)
            combined_features = tmp_features
        
        # 마스크 활성화
        mask = torch.sigmoid(combined_features)
        
        return mask, combined_features

class IntegratedROIDownsamplerNetwork(nn.Module):
    """통합된 ROI 다운샘플러 네트워크"""
    def __init__(self, scale_factor=2, in_channels=3, features=64, 
                 bg_color=[0.5, 0.5, 0.5], mask_threshold=0.5):
        super(IntegratedROIDownsamplerNetwork, self).__init__()
        
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.features = features
        self.bg_color = torch.tensor(bg_color).view(1, 3, 1, 1)
        self.mask_threshold = mask_threshold
        
        # 시퀀스 분석 모듈
        self.sequence_analyzer = SequenceAnalysisModule(in_channels, features)
        
        # 마스크 생성 모듈
        self.mask_generator = AdaptiveMaskGenerator(in_channels, features)
        
        # ROI 다운샘플러 (객체 영역)
        self.roi_downsampler = nn.Sequential(
            DownsamplerBlock(in_channels, features, scale_factor=1),
            ResBlock(features, features),
            DownsamplerBlock(features, features*2, scale_factor=scale_factor),
            ResBlock(features*2, features*2),
            ResBlock(features*2, features),
            nn.Conv2d(features, in_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # 배경 다운샘플러 (배경 영역)
        self.bg_downsampler = nn.Sequential(
            nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # 업샘플러 (테스트/추론 시 사용)
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # 비용 체적 (cost volume) 계산기
        self.cost_volume_network = nn.Sequential(
            nn.Conv2d(in_channels*2, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # 이전 프레임의 특징 및 마스크 저장
        self.prev_features = None
        self.prev_mask = None
    
    def forward(self, x, object_mask=None, prev_frames=None):
        batch_size, _, height, width = x.shape
        device = x.device
        
        # 배경 색상 장치 설정
        self.bg_color = self.bg_color.to(device)
        
        # 1. 시퀀스 분석 및 매개변수 예측
        pred_params, image_features = self.sequence_analyzer(x, prev_frames)
        
        # 2. 적응형 마스크 생성
        mask, mask_features = self.mask_generator(
            x, 
            object_mask=object_mask, 
            prev_mask=self.prev_mask, 
            prev_features=self.prev_features
        )
        
        # 3. 기본 다운샘플링 (배경, 낮은 비용)
        bg_downsampled = self.bg_downsampler(x)
        
        # 4. ROI 다운샘플링 (객체, 높은 비용)
        # 원본 크기에서 계산 후 다운샘플링
        roi_features = self.roi_downsampler[:3](x)  # 첫 번째 3개 레이어 통과
        
        # 다운샘플링된 크기로 변환
        roi_features = self.roi_downsampler[3:](roi_features)
        
        # 최종 ROI 다운샘플링 결과
        roi_downsampled = roi_features
        
        # 이미지 크기 계산
        ds_height = height // self.scale_factor
        ds_width = width // self.scale_factor
        
        # 마스크를 다운샘플링된 크기로 조정
        if mask.shape[2:] != (ds_height, ds_width):
            mask_downsampled = F.interpolate(mask, size=(ds_height, ds_width), mode='bilinear', align_corners=False)
        else:
            mask_downsampled = mask
        
        # 마스크 이진화
        binary_mask = (mask_downsampled > self.mask_threshold).float()
        
        # 5. 비용 체적 계산 (필요한 경우)
        if prev_frames is not None:
            # 비용 체적 계산
            prev_downsampled = self.bg_downsampler(prev_frames)
            cost_volume = self.cost_volume_network(torch.cat([roi_downsampled, prev_downsampled], dim=1))
            
            # 비용 체적을 최종 결과에 더함
            roi_downsampled = roi_downsampled + cost_volume * 0.2  # 가중치 0.2로 적용
        
        # 6. 마스크에 따라 ROI 영역과 배경 영역 결합
        # 배경 색상을 배치 크기에 맞게 확장
        expanded_bg_color = self.bg_color.expand(batch_size, -1, ds_height, ds_width)
        
        # (1-마스크) * ROI 다운샘플링 결과 + 마스크 * 배경 다운샘플링 결과
        downsampled = (1 - binary_mask) * roi_downsampled + binary_mask * bg_downsampled
        
        # 이전 프레임 특징 및 마스크 업데이트
        self.prev_features = mask_features.detach()
        self.prev_mask = mask.detach()
        
        return downsampled, mask

    def inference(self, x, object_mask=None, prev_frames=None):
        """추론 시 사용되는 인터페이스"""
        with torch.no_grad():
            downsampled, mask = self.forward(x, object_mask, prev_frames)
            return downsampled, mask

    def reset_temporal_memory(self):
        """시간적 메모리 초기화 (새 시퀀스 시작 시)"""
        self.prev_features = None
        self.prev_mask = None

def train_step(model, batch, optimizer, criterion, object_detector=None, device="cuda", 
               epoch=0, total_epochs=1, prev_frames=None):
    """통합 ROI 다운샘플러 학습 스텝"""
    
    # 배치 데이터 가져오기
    images = batch["image"].to(device)
    
    # 객체 탐지를 통한 마스크 생성 (필요한 경우)
    object_mask = None
    if object_detector is not None:
        mask_list = []
        for i in range(images.shape[0]):
            img = images[i].cpu().detach().permute(1, 2, 0).numpy() * 255
            img = img.astype(np.uint8)
            
            # 객체 탐지 및 세그멘테이션
            results = object_detector(img, verbose=False)[0]
            
            # 마스크 생성
            height, width = img.shape[:2]
            current_mask = torch.zeros((1, height, width), device=device)
            
            if hasattr(results, 'masks') and results.masks is not None and len(results.masks.data) > 0:
                for mask_data in results.masks.data:
                    mask = mask_data.cpu().numpy()
                    mask_tensor = torch.from_numpy(mask).to(device)
                    
                    if mask_tensor.shape[0] != height or mask_tensor.shape[1] != width:
                        mask_tensor = F.interpolate(
                            mask_tensor.unsqueeze(0).unsqueeze(0), 
                            size=(height, width), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    else:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    
                    # 마스크가 1이면 객체, 0이면 배경
                    current_mask = torch.maximum(current_mask, mask_tensor)
            
            mask_list.append(current_mask)
        
        # 배치 마스크로 변환
        object_mask = torch.stack(mask_list).to(device)
    
    # 이전 프레임 정보 (시퀀스 데이터인 경우)
    current_prev_frames = None
    if "sequence" in batch:
        sequences = batch["sequence"]
        # 중간 프레임을 기준으로 이전 프레임 사용
        sequence_length = sequences.shape[1]
        current_idx = sequence_length // 2
        
        # 이전 프레임이 있는 경우만 사용
        if current_idx > 0:
            current_prev_frames = sequences[:, current_idx-1].to(device)
    elif prev_frames is not None:
        current_prev_frames = prev_frames
    
    # 모델 순전파
    downsampled, generated_mask = model(images, object_mask, current_prev_frames)
    
    # 손실 계산
    loss, loss_components = criterion(
        original_img=images, 
        downsampled_img=downsampled, 
        object_mask=object_mask,
        mask=generated_mask,
        epoch=epoch
    )
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    # 그래디언트 클리핑
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), loss_components, downsampled, generated_mask, images 