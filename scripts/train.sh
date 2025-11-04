#!/bin/bash
# 학습 자동 재시작 스크립트
# 사용법: bash scripts/train.sh configs/ns_window-5_epoch-3.yaml

CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="configs/ns_window-5_epoch-3.yaml"
fi

# checkpoint 폴더 지정 (training_mode에 맞게 수정 필요)
CHECKPOINT_DIR="runs/checkpoints_ns"
LAST_CKPT=""

while true; do
    # 가장 최근 checkpoint 찾기
    if [ -d "$CHECKPOINT_DIR" ]; then
        LAST_CKPT=$(ls -t $CHECKPOINT_DIR/*.pth 2>/dev/null | head -n 1)
    fi

    if [ -n "$LAST_CKPT" ]; then
        echo "🔄 Resuming from checkpoint: $LAST_CKPT"
        python src/train.py --config $CONFIG --resume "$LAST_CKPT"
    else
        echo "🚀 Starting new training..."
        python src/train.py --config $CONFIG
    fi

    # python 종료 코드 확인
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed successfully!"
        break
    else
        echo "⚠️ Training stopped unexpectedly (exit code $EXIT_CODE). Restarting..."
        sleep 5  # 잠시 대기 후 재시작
    fi
done
