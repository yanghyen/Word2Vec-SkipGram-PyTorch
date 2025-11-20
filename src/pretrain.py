import os
import pickle
import re
from typing import Generator, List
from collections import Counter

import numpy as np
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
        return re.findall(r"\b\w+\b", text) 
    stopwords = set()

# -----------------------------
CORPUS_PATH = "data/pretrain/word2vec_corpus_hf_half.txt"
# TOKENIZED_SAVE_PATH = "data/pretrain/tokenized_corpus.txt"

STOPWORDS = set(stopwords.words('english')) if 'stopwords' in locals() and stopwords else set()

# -----------------------------
def clean_token(token: str):
    """전처리: URL, 숫자 제거, 특수문자 정리, 소문자화, 불용어 제거"""
    token = token.lower()

    token = re.sub(r'\d+', '', token)
    token = re.sub(r"[^a-z'-]", '', token)
    
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

# -----------------------------
def preprocess_text(text: str) -> list:
    """단일 문서 텍스트에 대해 전처리 및 토큰화를 수행합니다."""

    text = re.sub(r'==\s*(References|External links|See also|Notes|Sources)\s*==.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    tokens = word_tokenize(text)

    return preprocess_tokens(tokens)

# -----------------------------
def process_corpus_and_stream(path=CORPUS_PATH) -> Generator[List[str], None, None]:
    """
    원본 파일을 한 줄씩 읽어 문서('\n\n'으로 구분)를 재구성하고, 
    전처리 및 토큰화된 토큰 리스트(문장/문맥 단위)를 순차적으로 yield
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
                doce_text = " ".join(doc_buffer)
                
                tokens = preprocess_text(doce_text)
                
                if tokens:
                    yield tokens 
                    doc_count += 1
                    
                doc_buffer = []
                
                if doc_count % 100000 == 0 and doc_count > 0:
                    print(f"Processed {doc_count:,} documents so far...")
                    
        if doc_buffer:
            doce_text = " ".join(doc_buffer)
            tokens = preprocess_text(doce_text)
            if tokens:
                yield tokens 
                doc_count += 1
    print(f"\nProcessing complete. Total documents processd: {doc_count:,}")


def save_token_indices_to_binary(
    token_stream: Generator[List[str], None, None],
    word2idx: dict,
    save_path=TOKEN_INDICES_PATH
):
    print(f"Indexing corpus and saving to {save_path}...")

    all_indices = []
    total_tokens_count = 0
    
    for tokens in token_stream:
        indices = [word2idx[token] for token in tokens if token in word2idx]
        all_indices.extend(indices)
        
        total_tokens_count += len(indices)
        if total_tokens_count % 50000000 == 0 and total_tokens_count > 0:
            print(f"Tokens indexed so far: {total_tokens_count:,}")
    
    token_indices_array = np.array(all_indices, dtype=np.int32)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, token_indices_array)
    
    print(f"\n✅ Corpus indexing complete. Total indices: {len(token_indices_array):,}. Saved to {save_path}")

if __name__ == "__main__":
    
    # ----------------------------- 1. Vocab 구축 및 임시 파일 생성 (NS/HS 공통) -----------------------------
    try:
        # A. 원본 코퍼스를 읽어 토큰화된 내용을 임시 파일에 저장 (Vocab 구축용)
        print(f"Saving temporary tokenized corpus to {TOKENIZED_TRAIN_PATH} for vocab building...")
        total_temp_tokens = 0
        os.makedirs(os.path.dirname(TOKENIZED_TRAIN_PATH), exist_ok=True)
        with open(TOKENIZED_TRAIN_PATH, "w", encoding="utf-8") as f:
            temp_stream = process_corpus_and_stream(CORPUS_PATH)
            for tokens in temp_stream:
                f.write(" ".join(tokens) + "\n")
                total_temp_tokens += len(tokens)
        print(f"Temporary tokenized file created. Total tokens: {total_temp_tokens}")
        
        # B. 임시 파일로 Vocab 구축
        VOCAB_MIN_COUNT = 50 # config 값을 가정
        vocab, word2idx, idx2word, word_freq = build_vocab_stream(
            TOKENIZED_TRAIN_PATH,
            min_count=VOCAB_MIN_COUNT
        )
        
        # ----------------------------- 2. Vocab 파일 저장 (NS/HS 공통) -----------------------------
        # Vocab 파일 저장: train.py가 로드할 수 있도록 저장합니다. (여기서는 NS/HS 공통으로 사용한다고 가정)
        vocab_data = {"vocab": vocab, "word2idx": word2idx, "idx2word": idx2word, "word_freq": word_freq}
        vocab_filename = "vocab_data_3.pkl"  # train.py와 다른 스크립트들이 기대하는 파일명
        vocab_path = os.path.join("data/pretrain", vocab_filename)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab_data, f)
        print(f"✅ Final Vocab saved to {vocab_path}")
        
        # ----------------------------- 3. 학습 인덱스 생성 및 저장 -----------------------------
        # Vocab 구축을 위해 사용한 스트림은 소진되었으므로, 새 스트림 생성
        final_token_stream = process_corpus_and_stream(CORPUS_PATH) 
        save_token_indices_to_binary(final_token_stream, word2idx, TOKEN_INDICES_PATH)
        
        # ----------------------------- 4. 임시 파일 삭제 (유지) -----------------------------
        if os.path.exists(TOKENIZED_TRAIN_PATH):
            os.remove(TOKENIZED_TRAIN_PATH) # 👈 이 파일은 이제 필요 없으므로 삭제
            print(f"🧹 Removed temporary file: {TOKENIZED_TRAIN_PATH}")
            
    except FileNotFoundError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"치명적 오류: {e}")