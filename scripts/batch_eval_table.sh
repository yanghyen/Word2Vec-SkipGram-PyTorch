#!/bin/bash

# 배치 평가 테이블 생성 스크립트
# runs/checkpoints_ns와 runs/checkpoints_hs의 모든 .pth 파일들을 평가하여 하나의 CSV 테이블로 저장

# 사용법:
# scripts/batch_eval_table.sh [output_file]

OUTPUT_FILE=${1:-"results/batch_evaluation_table.csv"}

echo "🚀 배치 평가 시작..."
echo "📁 체크포인트 디렉토리: runs/checkpoints_ns, runs/checkpoints_hs"
echo "💾 출력 파일: $OUTPUT_FILE"

# 가상환경 활성화 (conda 환경 사용)
if command -v conda &> /dev/null; then
    echo "🐍 Conda 환경 활성화 중..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate word2vec
fi

# 결과 디렉토리 생성
mkdir -p $(dirname "$OUTPUT_FILE")

# Python 스크립트 실행
python src/batch_eval_table.py --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 배치 평가 완료!"
    echo "📊 결과 파일: $OUTPUT_FILE"
else
    echo "❌ 배치 평가 실패!"
    exit 1
fi
