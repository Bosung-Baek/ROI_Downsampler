#!/bin/bash

# 통합 ROI 다운샘플러 학습 스크립트

# 설정 변수
DATASET_PATH="datasets/imagenet_train"   # 데이터셋 경로
OUTPUT_DIR="output/roi_downsampler"      # 출력 디렉토리
EPOCHS=15                                # 학습 에폭 수
BATCH_SIZE=4                             # 배치 크기
SCALE_FACTOR=2                           # 다운샘플링 비율
PERCEPTUAL_WEIGHT=1.0                    # 지각적 손실 가중치
ROI_WEIGHT=5.0                           # ROI 손실 가중치
TEMPORAL_WEIGHT=2.0                      # 시간적 일관성 가중치
YOLO_VERSION="models/yolov8n-seg.pt"     # YOLO 세그멘테이션 모델

# 출력 디렉토리 생성
mkdir -p $OUTPUT_DIR

# 학습 실행
python train_roi_downsampler.py \
  --dataset_path $DATASET_PATH \
  --output_dir $OUTPUT_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --scale_factor $SCALE_FACTOR \
  --perceptual_weight $PERCEPTUAL_WEIGHT \
  --roi_weight $ROI_WEIGHT \
  --temporal_weight $TEMPORAL_WEIGHT \
  --yolo_version $YOLO_VERSION \
  --compare_masks 