# Word2Vec Skip-Gram 구현 (PyTorch)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Negative Sampling과 Hierarchical Softmax를 지원하는 Word2Vec Skip-Gram PyTorch 구현입니다.

## 설치

```bash
# 저장소 클론
git clone git@github.com:yanghyen/Word2Vec-SkipGram-PyTorch.git
cd Word2Vec_repo

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 통합 실행 스크립트 (권장)

전체 파이프라인을 자동으로 실행합니다 (위키피디아 다운로드 → 전처리 → 학습 → 평가):

```bash
# 전체 파이프라인 실행 (모든 config 파일로 학습 후 배치 평가)
./run.sh

# 특정 config로만 학습
./run.sh --config configs/ns_window-5_subsample-on_seed-42.yaml

# NS 모드만 학습
./run.sh --train-only --mode ns

# 환경 설정 및 데이터 준비만
./run.sh --setup-only

# 평가만 실행
./run.sh --eval-only
```

### 수동 실행

#### 학습

```bash
# 직접 실행
python src/train.py --config configs/ns_window-5_subsample-on_seed-42.yaml
```

#### 평가

```bash
bash scripts/eval.sh \
    configs/ns_window-5_subsample-on_seed-42.yaml \
    runs/checkpoints_ns/ns_window-5_sub-True_seed-42.pth \
    results/ns_window-5_sub-True_seed-42.csv
```

**출력 예시:**
```
📊 WordSim-353 Spearman: 0.6285
📘 SimLex-999 Spearman: 0.2639
👑 Google Analogy Accuracy: 0.3831
```

## 설정 파일

YAML 파일로 하이퍼파라미터를 설정합니다:

```yaml
# configs/ns_window-5_subsample-on_seed-42.yaml
training_mode: ns              # 'ns' 또는 'hs'
vocab_size: 30000
embedding_dim: 200
window_size: 5                 # 문맥 윈도우 크기
batch_size: 2048
lr: 0.001
epochs: 1
seed: 42
neg_sample_size: 5             # NS 전용
enable_subsampling: true       # 서브샘플링 활성화
subsample_t: 1e-3              # 서브샘플링 임계값
```

## 데이터 준비

### 자동 데이터 준비 (권장)

`run.sh`가 자동으로 처리합니다:

```bash
# 위키피디아 다운로드, 전처리, 평가 데이터셋 다운로드를 모두 자동으로 수행
./run.sh --setup-only
```

### 수동 데이터 준비

#### 위키피디아 코퍼스

```bash
# 1. Hugging Face에서 위키피디아 데이터셋 다운로드
python src/hugging.py

# 2. 코퍼스 파일 추출
python src/export_corpus.py

# 3. 전처리 (토큰화, 어휘 구축, 인덱스 생성)
python src/pretrain.py
```

#### 평가 데이터셋

```bash
bash scripts/download_all_data.sh
```

## 실험 결과

영어 위키피디아 2023년 1월 스냅샷 1/3 (약 220만 문서, 7억 토큰)로 학습:

| 모델 | WordSim-353 | SimLex-999 | Google Analogy |
|------|-------------|------------|----------------|
| **NS, W=2, Sub=T** | 0.644±0.002 | 0.304±0.005 | **0.462±0.003** |
| **NS, W=5, Sub=T** | 0.628±0.004 | 0.264±0.006 | 0.383±0.007 |
| **NS, W=5, Sub=F** | 0.630±0.009 | 0.259±0.005 | 0.378±0.003 |
| **HS, W=2, Sub=T** | 0.673±0.007 | 0.295±0.006 | 0.312±0.004 |
| **HS, W=5, Sub=T** | **0.699±0.007** | 0.285±0.006 | 0.344±0.001 |

*3개 시드 평균 성능 (±표준편차), 1 epoch 학습 기준*

## 프로젝트 구조

```
Word2Vec_repo/
├── configs/              # 실험 설정 파일 (YAML)
├── data/
│   ├── pretrain/         # 학습 데이터
│   └── word_similarity/  # 평가 데이터셋
├── src/
│   ├── model.py          # Skip-Gram 모델
│   ├── data.py           # 데이터 로더
│   ├── train.py          # 학습 스크립트
│   ├── eval.py           # 평가 스크립트
│   ├── pretrain.py       # 전처리 도구
│   ├── hugging.py        # 위키피디아 데이터셋 다운로드
│   ├── export_corpus.py  # 코퍼스 파일 추출
│   └── batch_eval_table.py  # 배치 평가 테이블 생성
├── scripts/              # 실행 스크립트
├── run.sh                # 통합 실행 스크립트
├── runs/                 # 학습 결과
└── results/              # 평가 결과
```

## 평가 메트릭

- **WordSim-353 / SimLex-999**: Spearman 순위 상관계수로 단어 유사도 평가
- **Google Analogy**: Top-1 정확도로 유추 태스크 평가 (`vec(b) - vec(a) + vec(c) ≈ vec(d)`)
