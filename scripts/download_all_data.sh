#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
ROOT=$(cd "$SCRIPT_DIR/.."; pwd)
DATA_DIR="$ROOT/data"
PRETRAIN_DIR="$DATA_DIR/pretrain"
WORDSIM_DIR="$DATA_DIR/word_similarity"

mkdir -p $PRETRAIN_DIR $WORDSIM_DIR 

# Wikipedia
cd $PRETRAIN_DIR

command -v aria2c >/dev/null 2>&1 || { echo >&2 "aria2c가 필요합니다. 설치 후 재실행하세요."; exit 1; }

cd $WORDSIM_DIR
if [ ! -f wordsim353.zip ]; then
    echo "Downloading WordSim-353..."
    aria2c -x 16 -s 16 http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip
    unzip wordsim353.zip
fi

if [ ! -f SimLex-999.txt ]; then
    echo "Downloading SimLex-999..."
    aria2c -x 16 -s 16 https://fh295.github.io/SimLex-999.zip -o SimLex-999.zip
    unzip SimLex-999.zip
fi

if [ ! -f google_analogy.zip ]; then
    echo "Downloading Google Analogy..."
    aria2c -x 16 -s 16 https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip -o google_analogy.zip
    unzip google_analogy.zip
fi

echo "All Word2Vec pretrain and evaluation datasets downloaded successfully!"