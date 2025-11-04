import os
import pickle
import re
from typing import Generator, List
from collections import Counter

import numpy as np
# NOTE: Assume build_vocab_stream, TOKEN_INDICES_PATH, TOKENIZED_TRAIN_PATH, word_stream_generator are imported from 'data'
from data import build_vocab_stream, TOKEN_INDICES_PATH, TOKENIZED_TRAIN_PATH, word_stream_generator 
import nltk
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True) 
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    print("NLTK is not installed. Using simple split() for tokenization.")
    def word_tokenize(text):
        return re.findall(r"\b\w+\b", text) # 최소한의 단어 경계로 분리
    stopwords = set()

# -----------------------------
CORPUS_PATH = "data/pretrain/word2vec_corpus_hf_half.txt"

STOPWORDS = set(stopwords.words('english')) if 'stopwords' in locals() and stopwords else set()
# -----------------------------

def clean_token(token: str):
    """전처리: URL, 숫자, 특수문자 정리, 소문자화, 불용어 제거"""
    token = token.lower()

    # 영문, 숫자, 하이픈, 어퍼스트로피만 허용
    token = re.sub(r"[^a-z0-9'-]", '', token)
    
    # 길이가 짧은 토큰이나 불용어 처리
    if not token or token in STOPWORDS or token.strip() == '' or len(token) < 2:
        return None
    return token

def preprocess_tokens(tokens: list):
    """토큰 리스트 전체 전처리"""
    cleaned_tokens = []
    for t in tokens:
        ct = clean_token(t)
        if ct:
            cleaned_tokens.append(ct)
    return cleaned_tokens

def preprocess_text(text: str) -> list:
    """단일 문서 텍스트에 대해 전처리 및 토큰화를 수행합니다."""
    # 참조 섹션 제거
    text = re.sub(r'==\s*(References|External links|See also|Notes|Sources)\s*==.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # URL 제거
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    tokens = word_tokenize(text)
    return preprocess_tokens(tokens)

def process_corpus_and_stream(path=CORPUS_PATH) -> Generator[List[str], None, None]:
    """
    원본 파일을 한 줄씩 읽어 문서를 재구성하고, 전처리 및 토큰화된 토큰 리스트를 순차적으로 yield합니다.
    (Vocab 구축용 임시 파일 생성 및 첫 번째 인덱싱 스트림에 사용)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found: {path}.")
    
    print(f"Starting streaming process from {path}.")
    
    doc_buffer = []
    doc_count = 0
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            
            if line:
                doc_buffer.append(line)
                
            if not line and doc_buffer:
                doc_text = " ".join(doc_buffer)
                tokens = preprocess_text(doc_text)
                
                if tokens:
                    yield tokens 
                    doc_count += 1
                doc_buffer = []
                
                if doc_count % 100000 == 0 and doc_count > 0:
                    print(f"Processed {doc_count:,} documents so far...")
                    
        # 마지막 문서 처리
        if doc_buffer:
            doc_text = " ".join(doc_buffer)
            tokens = preprocess_text(doc_text)
            if tokens:
                yield tokens 
                doc_count += 1
    print(f"\nProcessing complete. Total documents processd: {doc_count:,}")


def word_stream_generator_from_tokenized_file(path: str) -> Generator[List[str], None, None]:
    """
    ⭐ 최적화: 이미 토큰화되어 줄 단위로 저장된 임시 파일에서 토큰 스트림을 생성합니다.
    이 함수는 최종 인덱싱 시 I/O 및 CPU 연산을 절약합니다.
    """
    print(f"Starting stream from pre-tokenized file: {path}")
    doc_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # 이미 토큰화되어 공백으로 구분된 문자열이므로, split()만 사용
                tokens = line.split() 
                yield tokens
                doc_count += 1
                
                if doc_count % 500000 == 0 and doc_count > 0:
                    print(f"Streamed {doc_count:,} tokenized documents so far...")


def save_token_indices_to_binary(
    token_stream: Generator[List[str], None, None],
    word2idx: dict,
    save_path=TOKEN_INDICES_PATH
):
    """
    Vocab에 없는 단어는 모두 <unk> 인덱스로 치환하여 바이너리 파일로 저장합니다.
    """
    print(f"Indexing corpus and saving to {save_path}...")

    all_indices = []
    total_tokens_count = 0
    
    # <unk> 토큰의 인덱스 확인 (인덱스 0으로 가정)
    try:
        unk_idx = word2idx.get('<unk>', -1) 
        if unk_idx == -1:
             raise KeyError("<unk> token not found in word2idx. Check vocab building step.")
    except KeyError as e:
        print(f"❌ 오류: {e}. 인덱싱을 중단합니다.")
        return

    for tokens in token_stream:
        indices = []
        for token in tokens:
            # 토큰이 word2idx에 없으면 unk_idx를 사용합니다.
            idx = word2idx.get(token, unk_idx) 
            indices.append(idx)
            
        all_indices.extend(indices)
        
        total_tokens_count += len(indices)
        if total_tokens_count % 50000000 == 0 and total_tokens_count > 0:
            print(f"Tokens indexed so far: {total_tokens_count:,}")
    
    # NumPy 배열로 변환 및 저장 (바이너리 파일)
    token_indices_array = np.array(all_indices, dtype=np.int32)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, token_indices_array)
    
    print(f"\n✅ Corpus indexing complete. Total indices: {len(token_indices_array):,}. Saved to {save_path}")

# -----------------------------

if __name__ == "__main__":
    
    try:
        # ----------------------------- 1. Vocab 구축 및 임시 파일 생성 -----------------------------
        
        # A. 원본 코퍼스를 읽어 토큰화된 내용을 임시 파일에 저장 (Vocab 구축용)
        # 이 단계에서만 복잡한 전처리 과정을 수행합니다.
        print(f"Saving temporary tokenized corpus to {TOKENIZED_TRAIN_PATH} for vocab building...")
        total_temp_tokens = 0
        os.makedirs(os.path.dirname(TOKENIZED_TRAIN_PATH), exist_ok=True)
        with open(TOKENIZED_TRAIN_PATH, "w", encoding="utf-8") as f:
            temp_stream = process_corpus_and_stream(CORPUS_PATH)
            for tokens in temp_stream:
                f.write(" ".join(tokens) + "\n")
                total_temp_tokens += len(tokens)
        print(f"Temporary tokenized file created. Total tokens: {total_temp_tokens}")
        
        # B. 임시 파일로 Vocab 구축 (min_count=5 이하 단어는 제외)
        VOCAB_MIN_COUNT = 50
        vocab, word2idx, idx2word, word_freq = build_vocab_stream(
            TOKENIZED_TRAIN_PATH,
            min_count=VOCAB_MIN_COUNT
        )
        
        # ----------------------------- 2. <unk> 토큰 강제 추가 및 인덱스 재조정 -----------------------------
        
        new_vocab = {"<unk>": 0}
        new_word2idx = {"<unk>": 0}
        new_idx2word = {0: "<unk>"}
        new_word_freq = {"<unk>": 0} 
        
        current_idx = 1
        for word, count in sorted(vocab.items(), key=lambda item: item[1], reverse=True):
            if word not in new_word2idx: 
                new_vocab[word] = count
                new_word2idx[word] = current_idx
                new_idx2word[current_idx] = word
                new_word_freq[word] = count
                current_idx += 1
                
        vocab = new_vocab
        word2idx = new_word2idx
        idx2word = new_idx2word
        word_freq = new_word_freq

        # ----------------------------- 3. Vocab 파일 저장 (pkl 파일) -----------------------------
        vocab_data = {"vocab": vocab, "word2idx": word2idx, "idx2word": idx2word, "word_freq": word_freq}
        vocab_filename = "vocab_data_3.pkl" 
        vocab_path = os.path.join("data/pretrain", vocab_filename)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab_data, f)
        print(f"✅ Final Vocab (Size: {len(vocab):,}) saved to {vocab_path}")
        print(f"   <unk> 인덱스: {word2idx['<unk>']}, 다음 단어({idx2word[1]}): 1")
        
        # ----------------------------- 4. 학습 인덱스 생성 및 저장 (npy 바이너리 파일) -----------------------------
        # ⭐ 최적화 적용: 이미 토큰화된 임시 파일을 다시 읽어 스트림 생성
        final_token_stream = word_stream_generator_from_tokenized_file(TOKENIZED_TRAIN_PATH) 
        save_token_indices_to_binary(final_token_stream, word2idx, TOKEN_INDICES_PATH)
        
        # ----------------------------- 5. 임시 파일 삭제 -----------------------------
        if os.path.exists(TOKENIZED_TRAIN_PATH):
            os.remove(TOKENIZED_TRAIN_PATH) 
            print(f"🧹 Removed temporary file: {TOKENIZED_TRAIN_PATH}")
            
    except FileNotFoundError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"치명적 오류: {e}")
