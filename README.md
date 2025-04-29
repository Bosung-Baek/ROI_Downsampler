# 통합 ROI 다운샘플러 배포 패키지

이 패키지는 통합 ROI 다운샘플러 모델을 배포하기 위한 코드와 리소스를 포함하고 있습니다.

## 디렉토리 구조

```
deploy/
├── README.md                  # 이 설명 파일
├── requirements.txt           # 필요한 패키지 목록
├── roi_downsampler.py         # 통합 ROI 다운샘플러 네트워크 구현
├── blip_clip_loss.py          # 손실 함수 구현
├── train_roi_downsampler.py   # 모델 학습 스크립트
├── inference.py               # 추론 스크립트
├── train.sh                   # 학습 실행 스크립트 (실행 권한 있음)
├── inference.sh               # 추론 실행 스크립트 (실행 권한 있음)
├── models/                    # 사전 학습된 모델을 저장할 디렉토리
│   └── yolov8n-seg.pt         # YOLO 세그멘테이션 모델
└── output/                    # 결과물 저장 디렉토리
```

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 모델 학습

```bash
# 기본 학습 설정으로 실행
bash train.sh

# 또는 직접 스크립트 실행
python train_roi_downsampler.py \
  --dataset_path /경로/데이터셋 \
  --output_dir output/roi_downsampler \
  --epochs 15 \
  --batch_size 4 \
  --perceptual_weight 1.0 \
  --roi_weight 5.0 \
  --temporal_weight 2.0
```

### 모델 추론

```bash
# 기본 추론 설정으로 실행
bash inference.sh

# 또는 직접 스크립트 실행
python inference.py \
  --input_dir /경로/입력_시퀀스 \
  --output_dir /경로/출력_결과 \
  --checkpoint_path output/roi_downsampler/checkpoints/latest.pth \
  --seg_model_path models/yolov8n-seg.pt \
  --scale_factor 2
```

## 주요 매개변수

### 학습

- `--dataset_path`: 데이터셋 경로
- `--output_dir`: 출력 디렉토리
- `--epochs`: 학습 에폭 수
- `--batch_size`: 배치 크기
- `--perceptual_weight`: 지각적 손실 가중치
- `--roi_weight`: ROI 손실 가중치
- `--temporal_weight`: 시간적 일관성 손실 가중치
- `--scale_factor`: 다운샘플링 비율 (2, 4 등)

### 추론

- `--input_dir`: 입력 시퀀스 디렉토리
- `--output_dir`: 출력 디렉토리
- `--checkpoint_path`: 학습된 모델 체크포인트 경로
- `--seg_model_path`: 세그멘테이션 모델 경로
- `--scale_factor`: 다운샘플링 비율
- `--debug`: 디버그 모드 활성화 (추가 시각화)

## 구성 요소 설명

### 통합 ROI 다운샘플러 네트워크 (roi_downsampler.py)

시퀀스 분석, 적응형 마스크 생성, 시간적 일관성을 모두 통합한 엔드-투-엔드 네트워크를 구현합니다. 이 통합 모델은 입력 이미지 시퀀스를 분석하여 최적의 다운샘플링 파라미터를 예측하고, 객체 영역을 고품질로 보존하면서 배경은 효율적으로 압축합니다.

### 손실 함수 (blip_clip_loss.py)

모델 학습에 사용되는 다양한 손실 함수를 구현합니다. 통합 손실 함수는 지각적 손실(perceptual loss), ROI 영역 손실, 시간적 일관성 손실을 조합하여 최적의 다운샘플링 품질을 보장합니다.

### 학습 및 추론 스크립트

- `train_roi_downsampler.py`: 통합 모델을 학습하기 위한 스크립트
- `inference.py`: 학습된 모델을 사용하여 비디오 시퀀스를 처리하는 스크립트 