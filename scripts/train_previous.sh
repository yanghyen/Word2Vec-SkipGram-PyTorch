#!/bin/bash
# 학습 스크립트 실행
# 사용법: bash scripts/train.sh configs/ns_window-2_epoch-5.yaml

# scripts/train.sh configs/ns_window-2_epoch-5.yaml

CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="configs/ns_window-2_epoch-5.yaml"
fi

python src/train.py --config $CONFIG
