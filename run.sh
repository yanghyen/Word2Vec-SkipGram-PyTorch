#!/bin/bash

# =============================================================================
# Word2Vec 프로젝트 통합 실행 스크립트
# =============================================================================
# 이 스크립트는 Word2Vec 프로젝트의 전체 파이프라인을 실행합니다:
# 1. 환경 설정 및 의존성 설치
# 2. 위키피디아 코퍼스 다운로드 (Hugging Face) 및 전처리
# 3. 평가 데이터셋 다운로드 (WordSim-353, SimLex-999, Google Analogy)
# 4. 모든 config 파일로 모델 학습 (NS/HS 모드, 순차 실행)
# 5. 배치 평가 테이블 생성 (batch_eval_table.py)
# 6. 결과 분석 및 CSV 변환
#
# 사용법:
#   ./run.sh [옵션]
#
# 옵션:
#   --setup-only        환경 설정 및 데이터 다운로드만 수행
#   --train-only        학습만 수행
#   --eval-only         평가만 수행
#   --config CONFIG     특정 config 파일로 학습 (기본: 모든 config)
#   --mode MODE         특정 모드만 실행 (ns|hs|all, 기본: all)
#   --help             도움말 표시
# =============================================================================

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# 스크립트 디렉토리 및 루트 경로 설정
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
ROOT_DIR="$SCRIPT_DIR"
SRC_DIR="$ROOT_DIR/src"
SCRIPTS_DIR="$ROOT_DIR/scripts"
CONFIGS_DIR="$ROOT_DIR/configs"
DATA_DIR="$ROOT_DIR/data"
RESULTS_DIR="$ROOT_DIR/results"
RUNS_DIR="$ROOT_DIR/runs"

# 기본 설정
SETUP_ONLY=false
TRAIN_ONLY=false
EVAL_ONLY=false
SPECIFIC_CONFIG=""
MODE="all"  # ns, hs, all

# 도움말 함수
show_help() {
    echo "Word2Vec 프로젝트 통합 실행 스크립트"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  --setup-only        환경 설정 및 데이터 다운로드만 수행"
    echo "  --train-only        학습만 수행"
    echo "  --eval-only         평가만 수행"
    echo "  --config CONFIG     특정 config 파일로 학습"
    echo "  --mode MODE         특정 모드만 실행 (ns|hs|all, 기본: all)"
    echo "  --help             이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0                                    # 전체 파이프라인 실행"
    echo "  $0 --setup-only                      # 환경 설정만"
    echo "  $0 --train-only --mode ns            # NS 모드 학습만"
    echo "  $0 --config configs/ns_window-5_subsample-on_seed-42.yaml"
    echo "  $0 --eval-only                       # 평가만"
}

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --config)
            SPECIFIC_CONFIG="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 모드 검증
if [[ "$MODE" != "ns" && "$MODE" != "hs" && "$MODE" != "all" ]]; then
    log_error "잘못된 모드: $MODE (ns, hs, all 중 선택)"
    exit 1
fi

# 환경 설정 함수
setup_environment() {
    log_header "환경 설정"
    
    # Python 버전 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python3가 설치되어 있지 않습니다."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python 버전: $PYTHON_VERSION"
    
    # 가상환경 활성화 (conda 환경)
    if command -v conda &> /dev/null; then
        log_info "Conda 환경 활성화 중..."
        source $(conda info --base)/etc/profile.d/conda.sh
        
        # word2vec 환경이 있는지 확인
        if conda env list | grep -q "word2vec"; then
            conda activate word2vec
            log_success "Conda 환경 'word2vec' 활성화됨"
        else
            log_warning "Conda 환경 'word2vec'가 없습니다. 기본 환경을 사용합니다."
        fi
    fi
    
    # 의존성 설치
    if [ -f "$ROOT_DIR/requirements.txt" ]; then
        log_info "의존성 패키지 설치 중..."
        pip install -r "$ROOT_DIR/requirements.txt"
        log_success "의존성 패키지 설치 완료"
    else
        log_warning "requirements.txt 파일이 없습니다."
    fi
    
    # 필요한 디렉토리 생성
    mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$RUNS_DIR"
    mkdir -p "$DATA_DIR/pretrain" "$DATA_DIR/word_similarity"
    mkdir -p "$RUNS_DIR/checkpoints_ns" "$RUNS_DIR/checkpoints_hs"
    mkdir -p "$RUNS_DIR/eval/go"
    mkdir -p "$RUNS_DIR/metrics"
    
    log_success "환경 설정 완료"
}

# 위키피디아 데이터 다운로드 함수
download_wikipedia() {
    log_header "위키피디아 데이터 다운로드"
    
    PRETRAIN_DIR="$DATA_DIR/pretrain"
    CORPUS_FILE="$PRETRAIN_DIR/word2vec_corpus_hf_half.txt"
    VOCAB_FILE="$PRETRAIN_DIR/vocab_data_3.pkl"
    TOKEN_INDICES_FILE="$PRETRAIN_DIR/token_indices_3.npy"
    
    # 이미 전처리된 데이터가 있으면 스킵
    if [ -f "$VOCAB_FILE" ] && [ -f "$TOKEN_INDICES_FILE" ]; then
        log_info "전처리된 학습 데이터가 이미 존재합니다."
        log_info "  - Vocab: $VOCAB_FILE"
        log_info "  - Token Indices: $TOKEN_INDICES_FILE"
        log_success "학습 데이터 준비 완료 (스킵)"
        return 0
    fi
    
    # 1. Hugging Face에서 위키피디아 데이터셋 다운로드
    if [ ! -d "$PRETRAIN_DIR/huggingface_cache" ]; then
        log_info "Hugging Face에서 위키피디아 데이터셋 다운로드 중..."
        log_info "이 작업은 시간이 오래 걸릴 수 있습니다..."
        
        if [ -f "$SRC_DIR/hugging.py" ]; then
            python3 "$SRC_DIR/hugging.py"
            log_success "위키피디아 데이터셋 다운로드 완료"
        else
            log_error "hugging.py 파일을 찾을 수 없습니다: $SRC_DIR/hugging.py"
            return 1
        fi
    else
        log_info "위키피디아 데이터셋 캐시가 이미 존재합니다."
    fi
    
    # 2. 코퍼스 파일 추출
    if [ ! -f "$CORPUS_FILE" ]; then
        log_info "코퍼스 파일 추출 중..."
        
        if [ -f "$SRC_DIR/export_corpus.py" ]; then
            # export_corpus.py의 하드코딩된 경로를 수정하기 위해 임시로 수정
            python3 "$SRC_DIR/export_corpus.py"
            log_success "코퍼스 파일 추출 완료: $CORPUS_FILE"
        else
            log_error "export_corpus.py 파일을 찾을 수 없습니다: $SRC_DIR/export_corpus.py"
            return 1
        fi
    else
        log_info "코퍼스 파일이 이미 존재합니다: $CORPUS_FILE"
    fi
    
    # 3. 전처리 (Vocab 구축 및 Token Indices 생성)
    if [ ! -f "$VOCAB_FILE" ] || [ ! -f "$TOKEN_INDICES_FILE" ]; then
        log_info "코퍼스 전처리 중 (Vocab 구축 및 Token Indices 생성)..."
        log_info "이 작업은 시간이 오래 걸릴 수 있습니다..."
        
        if [ -f "$SRC_DIR/pretrain.py" ]; then
            python3 "$SRC_DIR/pretrain.py"
            log_success "전처리 완료"
        else
            log_error "pretrain.py 파일을 찾을 수 없습니다: $SRC_DIR/pretrain.py"
            return 1
        fi
    else
        log_info "전처리된 파일들이 이미 존재합니다."
    fi
    
    # 최종 확인
    if [ -f "$VOCAB_FILE" ] && [ -f "$TOKEN_INDICES_FILE" ]; then
        log_success "학습 데이터 준비 완료"
        log_info "  - Vocab: $VOCAB_FILE"
        log_info "  - Token Indices: $TOKEN_INDICES_FILE"
    else
        log_error "전처리 실패: 필요한 파일이 생성되지 않았습니다."
        return 1
    fi
}

# 평가 데이터셋 다운로드 함수
download_eval_data() {
    log_header "평가 데이터셋 다운로드"
    
    if [ -f "$SCRIPTS_DIR/download_all_data.sh" ]; then
        log_info "평가 데이터셋 다운로드 스크립트 실행 중..."
        bash "$SCRIPTS_DIR/download_all_data.sh"
        log_success "평가 데이터셋 다운로드 완료"
    else
        log_warning "데이터 다운로드 스크립트가 없습니다: $SCRIPTS_DIR/download_all_data.sh"
        
        # 수동으로 필요한 데이터 확인
        log_info "필요한 평가 데이터 파일들을 확인 중..."
        
        REQUIRED_FILES=(
            "$DATA_DIR/word_similarity/combined.csv"
            "$DATA_DIR/word_similarity/SimLex-999/SimLex-999.txt"
            "$DATA_DIR/word_similarity/word2vec/trunk/questions-words.txt"
        )
        
        MISSING_FILES=()
        for file in "${REQUIRED_FILES[@]}"; do
            if [ ! -f "$file" ]; then
                MISSING_FILES+=("$file")
            fi
        done
        
        if [ ${#MISSING_FILES[@]} -gt 0 ]; then
            log_warning "다음 평가 데이터 파일들이 누락되었습니다:"
            for file in "${MISSING_FILES[@]}"; do
                echo "  - $file"
            done
            log_warning "수동으로 데이터를 준비하거나 download_all_data.sh 스크립트를 확인하세요."
        else
            log_success "모든 평가 데이터 파일이 존재합니다."
        fi
    fi
}

# 데이터 다운로드 함수 (통합)
download_data() {
    # 위키피디아 학습 데이터 다운로드 및 전처리
    download_wikipedia
    
    # 평가 데이터셋 다운로드
    download_eval_data
}

# 학습 함수
train_models() {
    log_header "모델 학습"
    
    if [ -n "$SPECIFIC_CONFIG" ]; then
        # 특정 config 파일로 학습
        if [ ! -f "$SPECIFIC_CONFIG" ]; then
            log_error "Config 파일이 존재하지 않습니다: $SPECIFIC_CONFIG"
            exit 1
        fi
        
        log_info "특정 config로 학습 시작: $SPECIFIC_CONFIG"
        
        if [ -f "$SCRIPTS_DIR/train.sh" ]; then
            bash "$SCRIPTS_DIR/train.sh" "$SPECIFIC_CONFIG"
        else
            python3 "$SRC_DIR/train.py" --config "$SPECIFIC_CONFIG"
        fi
        
        log_success "학습 완료: $SPECIFIC_CONFIG"
    else
        # 모든 config 파일로 학습
        CONFIG_PATTERN=""
        case $MODE in
            "ns")
                CONFIG_PATTERN="ns_*.yaml"
                ;;
            "hs")
                CONFIG_PATTERN="hs_*.yaml"
                ;;
            "all")
                CONFIG_PATTERN="*.yaml"
                ;;
        esac
        
        log_info "모드 '$MODE'에 해당하는 config 파일들로 학습 시작..."
        
        CONFIG_FILES=($(find "$CONFIGS_DIR" -name "$CONFIG_PATTERN" -type f | sort))
        
        if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
            log_warning "해당하는 config 파일이 없습니다: $CONFIGS_DIR/$CONFIG_PATTERN"
            return
        fi
        
        log_info "총 ${#CONFIG_FILES[@]}개의 config 파일을 찾았습니다."
        
        for config_file in "${CONFIG_FILES[@]}"; do
            log_info "학습 시작: $(basename "$config_file")"
            
            if [ -f "$SCRIPTS_DIR/train.sh" ]; then
                bash "$SCRIPTS_DIR/train.sh" "$config_file"
            else
                python3 "$SRC_DIR/train.py" --config "$config_file"
            fi
            
            log_success "학습 완료: $(basename "$config_file")"
        done
    fi
    
    log_success "모든 모델 학습 완료"
}

# 평가 함수
evaluate_models() {
    log_header "모델 평가"
    
    # 배치 평가 테이블 생성
    if [ -f "$SRC_DIR/batch_eval_table.py" ]; then
        log_info "배치 평가 테이블 생성 중..."
        
        OUTPUT_FILE="$RESULTS_DIR/batch_evaluation_table.csv"
        
        if [ -f "$SCRIPTS_DIR/batch_eval_table.sh" ]; then
            bash "$SCRIPTS_DIR/batch_eval_table.sh" "$OUTPUT_FILE"
        else
            python3 "$SRC_DIR/batch_eval_table.py" --output "$OUTPUT_FILE"
        fi
        
        log_success "배치 평가 완료: $OUTPUT_FILE"
    else
        log_warning "배치 평가 스크립트가 없습니다: $SRC_DIR/batch_eval_table.py"
    fi
    
    # 개별 모델 평가 (선택적)
    if [ -f "$SCRIPTS_DIR/eval.sh" ]; then
        log_info "개별 모델 평가 스크립트가 있습니다: $SCRIPTS_DIR/eval.sh"
        log_info "필요시 수동으로 실행하세요."
    fi
}

# 결과 분석 함수
analyze_results() {
    log_header "결과 분석"
    
    # 결과 분석 스크립트 실행
    if [ -f "$SRC_DIR/analyze_results.py" ]; then
        log_info "결과 분석 중..."
        python3 "$SRC_DIR/analyze_results.py"
        log_success "결과 분석 완료"
    fi
    
    # CSV 변환 스크립트 실행
    if [ -f "$SRC_DIR/analyze_to_csv.py" ]; then
        log_info "CSV 변환 중..."
        python3 "$SRC_DIR/analyze_to_csv.py"
        log_success "CSV 변환 완료"
    fi
    
    # 결과 파일들 표시
    log_info "생성된 결과 파일들:"
    if [ -d "$RESULTS_DIR" ]; then
        find "$RESULTS_DIR" -name "*.csv" -type f | while read file; do
            echo "  📊 $file"
        done
    fi
}

# 메인 실행 함수
main() {
    log_header "Word2Vec 프로젝트 실행 시작"
    
    # 현재 디렉토리 확인
    log_info "작업 디렉토리: $ROOT_DIR"
    log_info "실행 모드: $MODE"
    
    if [ "$SETUP_ONLY" = true ]; then
        setup_environment
        download_data
        log_success "환경 설정 및 데이터 다운로드 완료"
        return
    fi
    
    if [ "$TRAIN_ONLY" = false ] && [ "$EVAL_ONLY" = false ]; then
        # 전체 파이프라인 실행
        setup_environment
        download_data
        train_models
        evaluate_models
        analyze_results
    elif [ "$TRAIN_ONLY" = true ]; then
        setup_environment
        train_models
    elif [ "$EVAL_ONLY" = true ]; then
        setup_environment
        evaluate_models
        analyze_results
    fi
    
    log_success "모든 작업 완료!"
    
    # 최종 결과 요약
    log_header "실행 결과 요약"
    
    if [ -d "$RUNS_DIR" ]; then
        CHECKPOINT_COUNT=$(find "$RUNS_DIR" -name "*.pth" -type f | wc -l)
        log_info "생성된 체크포인트: ${CHECKPOINT_COUNT}개"
    fi
    
    if [ -d "$RESULTS_DIR" ]; then
        RESULT_COUNT=$(find "$RESULTS_DIR" -name "*.csv" -type f | wc -l)
        log_info "생성된 결과 파일: ${RESULT_COUNT}개"
    fi
    
    log_info "자세한 결과는 다음 디렉토리에서 확인하세요:"
    echo "  📁 체크포인트: $RUNS_DIR"
    echo "  📊 결과: $RESULTS_DIR"
}

# 스크립트 실행
main "$@"
