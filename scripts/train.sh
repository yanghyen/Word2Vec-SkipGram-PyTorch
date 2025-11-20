#!/bin/bash
# 학습 실행 스크립트
# 사용법: bash scripts/train.sh configs/ns_window-5_epoch-3.yaml

CONFIG=$1

if [ -z "$CONFIG" ]; then
    echo "❌ Config 파일을 지정해주세요."
    echo "사용법: bash scripts/train.sh <config_file>"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ Config 파일을 찾을 수 없습니다: $CONFIG"
    exit 1
fi

echo "🚀 학습 시작: $CONFIG"
python src/train.py --config "$CONFIG"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 학습 완료!"
else
    echo "❌ 학습 실패 (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
