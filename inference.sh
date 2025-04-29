#!/bin/bash

# 통합 ROI 다운샘플러 추론 스크립트

# 설정 변수
INPUT_DIR="datasets/imagenet_original"    # 입력 시퀀스 디렉토리
OUTPUT_DIR="inference_results"            # 출력 디렉토리
CHECKPOINT="output/roi_downsampler/checkpoints/latest.pth"  # 모델 체크포인트
SEG_MODEL="models/yolov8n-seg.pt"         # 세그멘테이션 모델
SCALE_FACTOR=2                            # 다운샘플링 비율

# 추론 실행
python inference.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --checkpoint_path $CHECKPOINT \
  --seg_model_path $SEG_MODEL \
  --scale_factor $SCALE_FACTOR 