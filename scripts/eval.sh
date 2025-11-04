#!/bin/bash

# 사용법:
# scripts/eval.sh configs/ns_window-2_epoch-5.yaml runs/checkpoints/window-2_epoch-5.pth runs/checkpoints_ns/ns_window-2_epoch-5.csv

CONFIG=$1
CHECKPOINT=$2
SAVE_CSV=$3  # 선택적, 없으면 CSV 저장 안 함

# 고정 경로
COMBINED_CSV="data/word_similarity/combined.csv"
SIMLEX_CSV="data/word_similarity/SimLex-999/SimLex-999.txt"
ANALOGY_CSV="data/word_similarity/word2vec/trunk/questions-words.txt"

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./eval.sh <config.yaml> <checkpoint.pth> [save_csv_path]"
    exit 1
fi

# 기본 실행 명령 (positional arguments로)
CMD="python src/eval.py $CONFIG $CHECKPOINT $COMBINED_CSV $SIMLEX_CSV $ANALOGY_CSV"

# save_csv 옵션이 주어지면 추가
if [ ! -z "$SAVE_CSV" ]; then
    CMD="$CMD --save_csv $SAVE_CSV"
fi

# 실행
eval $CMD
#!/bin/bash

# 사용법:
# scripts/eval.sh configs/ns_window-2_epoch-5.yaml runs/checkpoints_ns/ns_window-2_epoch-5.pth runs/checkpoints_ns/ns_window-2_epoch-5.csv


CONFIG=$1
CHECKPOINT=$2
SAVE_CSV=$3  # 선택적, 없으면 CSV 저장 안 함

# 고정 경로
COMBINED_CSV="data/word_similarity/combined.csv"
SIMLEX_CSV="data/word_similarity/SimLex-999/SimLex-999.txt"
ANALOGY_CSV="data/word_similarity/word2vec/trunk/questions-words.txt"

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./eval.sh <config.yaml> <checkpoint.pth> [save_csv_path]"
    exit 1
fi

# 기본 실행 명령 (positional arguments로)
CMD="python src/eval.py $CONFIG $CHECKPOINT $COMBINED_CSV $SIMLEX_CSV $ANALOGY_CSV"

# save_csv 옵션이 주어지면 추가
if [ ! -z "$SAVE_CSV" ]; then
    CMD="$CMD --save_csv $SAVE_CSV"
fi

# 실행
eval $CMD
