# src/data.py
from collections import Counter
import random
import csv
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
import pickle
from typing import Generator, List, Dict, Any

import numpy as np
from huffman_tree import HuffmanTree
import xml.etree.ElementTree as ET

def subsample_text(text, t=1e-3): 
    counter = Counter(text)
    total_count = len(text)
    freqs = {word: count / total_count for word, count in counter.items()}
    
    subsampled = []
    for word in text:
        f = freqs[word]
        p_drop = 1 - ((t / f) ** 0.5)
        p_drop = max(0, p_drop) 
        
        if random.random() > p_drop: 
            subsampled.append(word)
    return subsampled

TOKENIZED_TRAIN_PATH = "data/pretrain/tokenized_corpus.txt"
TOKEN_INDICES_PATH = "data/pretrain/token_indices_3.npy"

def word_stream_generator(file_path) -> Generator[List[str], None, None]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tokenized corpus file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for token in line.strip().split():
                yield token
                
def build_vocab_stream(file_path, min_count=50):
    vocab_counter = Counter()
    print(f"Building vocab from stream: {file_path} (min_counter={min_count})")
    
    for token in word_stream_generator(file_path):
        vocab_counter[token] += 1
        
    vocab = {word: count for word, count in vocab_counter.items() if count >= min_count}
    word2idx = {word: i for i, word in enumerate(vocab.keys())}
    idx2word = {i: word for word, i in word2idx.items()}
    
    print(f"Built vocab: {len(vocab)} words (min_count={min_count})")
    word_freq = {word: vocab_counter[word] for word in vocab}
    
    return vocab, word2idx, idx2word, word_freq

def load_wordsim353(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            w1, w2, sim = row[0].lower(), row[1].lower(), float(row[2])
            pairs.append((w1, w2, sim))
    return pairs

def load_simlex999(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            w1, w2, sim = row['word1'].lower(), row['word2'].lower(), float(row['SimLex999'])
            pairs.append((w1, w2, sim))
    return pairs

def load_google_analogy(path):
    analogies = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':') or len(line) == 0:
                continue
            words = line.lower().split()
            if len(words) == 4:
                analogies.append(tuple(words))
    return analogies

class SkipGramNSIterableDataset(IterableDataset):
    def __init__(self, file_path, word2idx, word_freq, device="cuda", neg_sample_size=5, window_size=2, subsample_t=1e-3):
        super().__init__()
        self.file_path = file_path
        self.word2idx = word2idx
        self.window_size = window_size
        self.neg_sample_size = neg_sample_size
        self.subsample_t = subsample_t
        self.vocab_size = len(word2idx)
        self.idx2word = {i: w for w, i in word2idx.items()} # 편의를 위해 추가
        
        # Negative Sampling 확률 계산: min_count를 통과한 단어들의 빈도 사용
        freqs_list = [word_freq.get(self.idx2word.get(i), 0) for i in range(self.vocab_size)] 
        self.freqs_for_neg = torch.tensor(freqs_list, dtype=torch.float)
        self.freqs_for_neg[self.freqs_for_neg == 0] = 1e-30 # 0 방지
        self.sample_probs = self.freqs_for_neg.pow(0.75) / self.freqs_for_neg.pow(0.75).sum()
        # self.sample_probs = self.sample_probs.to(device)
        
        # Subsampling 확률 계산
        self.total_count = sum(word_freq.values())
        self.freqs = {word: count / self.total_count for word, count in word_freq.items()}
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Binary index file not found: {self.file_path}. Run preprocessing first!")
        
        try:
            self.token_indices = np.load(self.file_path, mmap_mode='r')
            self.total_tokens = len(self.token_indices)
            print(f"Loaded token indices (mmap): Total tokens = {self.total_tokens:,}")
        except Exception as e:
            raise RuntimeError(f"Error loading token indices file via mmap: {e}")
        
    
    def __iter__(self):
        """학습 쌍 (center, context, neg_samples)을 yield"""
        
        # 🟢 [수정] 워커별 데이터 분할 (mmap 배열 인덱스를 사용하여 병렬 처리)
        worker_info = torch.utils.data.get_worker_info()
        overlap = self.window_size
        
        if worker_info is None:
            start_idx = 0
            end_idx = self.total_tokens
        else:
            per_worker = self.total_tokens // worker_info.num_workers
            
            start_idx = worker_info.id * per_worker
            end_idx = start_idx + per_worker
            
            if worker_info.id > 0:
                start_idx = max(0, start_idx - overlap)
            if worker_info.id < worker_info.num_workers - 1:
                end_idx = min(self.total_tokens, end_idx + overlap)     
            else:
                end_idx = self.total_tokens
        current_idx = start_idx + (overlap if worker_info and worker_info.id > 0 else 0)
        actual_end = end_idx - (overlap if worker_info and worker_info.id < worker_info.num_workers - 1 else 0)
        
        while current_idx < end_idx:
            # 1. 중심 단어 설정 및 서브샘플링
            center_idx = self.token_indices[current_idx] # mmap 배열에서 인덱스 접근
            center_token = self.idx2word[center_idx]
            
            # 서브샘플링 확률
            f = self.freqs.get(center_token, 0)
            p_drop = 1 - ((self.subsample_t / f) ** 0.5) if f > 0 else 1
            if random.random() < p_drop: 
                current_idx += 1 # 드롭된 경우에도 인덱스 증가
                continue 

            # 2. 가변 윈도우 설정
            actual_window = random.randint(1, self.window_size)
            
            # 3. 주변 단어(Context) 선택
            # 문맥 인덱스 범위 계산 (현재 워커의 범위(start_idx, end_idx)를 벗어나지 않도록 제한)
            left_context_start = max(0, current_idx - actual_window)
            right_context_end = min(self.total_tokens, current_idx + actual_window + 1)
            
            for context_token_idx in range(left_context_start, right_context_end):
                if context_token_idx == current_idx:
                    continue
                    
                context_idx = self.token_indices[context_token_idx]

                # 4. 네거티브 샘플링 (변경 없음)
                neg_samples = torch.multinomial(
                    self.sample_probs,
                    self.neg_sample_size,
                    replacement=True
                )
                
                # 5. 학습 쌍 Yield
                yield torch.tensor(center_idx, dtype=torch.long), \
                      torch.tensor(context_idx, dtype=torch.long), \
                      neg_samples
                      
            current_idx += 1 # 중심 단어 인덱스 증가 
                          
                          
class SkipGramNSDataset(Dataset):
    def __init__(self, tokens, word2idx, vocab, neg_sample_size=5, window_size=2):
        
        self.tokens = tokens  
        self.word2idx = word2idx
        self.window_size = window_size
        self.neg_sample_size = neg_sample_size
        
        token_counter = Counter(self.tokens)
        
        idx_to_word_list = sorted(word2idx.items(), key=lambda item: item[1])
        
        freqs_list = []
        for word, idx in idx_to_word_list:
            freqs_list.append(token_counter.get(word, 0))
            
        freqs = torch.tensor(freqs_list, dtype=torch.float)
        freqs[freqs == 0] = 1e-30
        self.sample_probs = freqs.pow(0.75) / freqs.pow(0.75).sum()
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        max_retries = 10
        for _ in range(max_retries):
            try:
                center_token = self.tokens[idx]
                center_idx = torch.tensor(self.word2idx[center_token], dtype=torch.long)

                window = random.randint(1, self.window_size)
                start = max(0, idx - window)
                end = min(len(self.tokens), idx + window + 1)
                
                context_candidates = list(range(start, end))
                if idx in context_candidates:
                    context_candidates.remove(idx)

                if not context_candidates:
                    j = random.choice([k for k in range(max(0, idx - 1), min(len(self.tokens), idx + 2)) if k != idx])
                else:
                    j = random.choice(context_candidates)

                context_token = self.tokens[j]
                context_idx = torch.tensor(self.word2idx[context_token], dtype=torch.long)

                neg_samples = torch.multinomial(
                    self.sample_probs,
                    self.neg_sample_size,
                    replacement=True
                )
                
                return center_idx, context_idx, neg_samples
                
            except KeyError:
                idx = random.randint(0, len(self.tokens) - 1)
                continue
                
        raise RuntimeError("Failed to sample a valid token after multiple retries.")

class SkipGramHSDataset(Dataset):
    def __init__(self, tokens, word2idx, window_size=2):
        self.tokens = tokens  
        self.word2idx = word2idx
        self.window_size = window_size

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        center_token = self.tokens[idx]
        center_idx = torch.tensor(self.word2idx[center_token], dtype=torch.long)

        window = random.randint(1, self.window_size)
        start = max(0, idx - window)
        end = min(len(self.tokens), idx + window + 1)
        
        context_candidates = list(range(start, end))
        if idx in context_candidates:
          context_candidates.remove(idx)
        
        j = random.choice(context_candidates) if context_candidates else idx
        
        target_token = self.tokens[j]
        target_idx = torch.tensor(self.word2idx[target_token], dtype=torch.long)
        
        return center_idx, target_idx
    
class SkipGramHSIterableDataset(IterableDataset):
    """
    Hierarchical Softmax 학습을 위한 Skip-Gram Iterable Dataset.
    mmap된 토큰 인덱스 파일을 스트리밍하여 (center_idx, path, code) 쌍을 생성합니다.
    """
    def __init__(self, file_path, word2idx, word_freq, path_table, code_table, window_size=2, subsample_t=1e-3):
        super().__init__()
        self.file_path = file_path
        self.word2idx = word2idx
        self.window_size = window_size
        self.subsample_t = subsample_t
        self.vocab_size = len(word2idx)
        self.idx2word = {i: w for w, i in word2idx.items()}
        
        # Huffman Tree 경로/코드 테이블
        if path_table is None or code_table is None:
             raise ValueError("path_table and code_table must be provided for Hierarchical Softmax.")
        self.path_table = path_table # target index의 부모 노드 인덱스 리스트 (경로)
        self.code_table = code_table # target index의 이진 코드 리스트
        
        # Subsampling 확률 계산 (SkipGramNSIterableDataset과 동일)
        self.total_count = sum(word_freq.values())
        self.freqs = {word: count / self.total_count for word, count in word_freq.items()}
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Binary index file not found: {self.file_path}. Run preprocessing first!")
        
        try:
            # mmap_mode='r'로 메모리 맵핑하여 대용량 파일 처리 및 워커 간 공유
            self.token_indices = np.load(self.file_path, mmap_mode='r')
            self.total_tokens = len(self.token_indices)
            print(f"Loaded token indices (mmap) for HS: Total tokens = {self.total_tokens:,}")
        except Exception as e:
            raise RuntimeError(f"Error loading token indices file via mmap: {e}")

    def __iter__(self):
        """학습 쌍 (center_idx, target_path, target_code)을 yield"""
        
        # 🟢 워커별 데이터 분할 (SkipGramNSIterableDataset과 동일한 로직)
        worker_info = torch.utils.data.get_worker_info()
        overlap = self.window_size # 경계에서 context를 놓치지 않기 위함

        if worker_info is None:
            start_idx = 0
            end_idx = self.total_tokens
        else:
            per_worker = self.total_tokens // worker_info.num_workers
            start_idx = worker_info.id * per_worker
            end_idx = start_idx + per_worker
            
            # 워커 경계에서 윈도우 크기만큼 오버랩
            if worker_info.id > 0:
                start_idx = max(0, start_idx - overlap)
            if worker_info.id < worker_info.num_workers - 1:
                end_idx = min(self.total_tokens, end_idx + overlap) 
            else:
                end_idx = self.total_tokens # 마지막 워커는 끝까지
                
        current_idx = start_idx + (overlap if worker_info and worker_info.id > 0 else 0) # 실제 시작 인덱스
        
        while current_idx < end_idx:
            # 1. 중심 단어 설정 및 서브샘플링 (SkipGramNSIterableDataset과 동일)
            center_idx = self.token_indices[current_idx] 
            center_token = self.idx2word[center_idx]
            
            # 서브샘플링 확률
            f = self.freqs.get(center_token, 0)
            p_drop = 1 - ((self.subsample_t / f) ** 0.5) if f > 0 else 1
            if random.random() < p_drop: 
                current_idx += 1 
                continue 

            # 2. 가변 윈도우 설정
            actual_window = random.randint(1, self.window_size)
            
            # 3. 주변 단어(Target) 선택
            # 문맥 인덱스 범위 계산 (현재 워커의 범위(start_idx, end_idx)를 벗어나지 않도록 제한)
            left_context_start = max(start_idx, current_idx - actual_window)
            right_context_end = min(end_idx, current_idx + actual_window + 1)
            
            for target_token_idx in range(left_context_start, right_context_end):
                if target_token_idx == current_idx:
                    continue
                
                target_idx = self.token_indices[target_token_idx]

                # 4. Hierarchical Softmax 경로 및 코드 가져오기
                # target_idx는 단어 인덱스이며, path_table/code_table의 인덱스로 사용됨
                # path_table[target_idx] -> 부모 노드 인덱스 리스트 (경로)
                # code_table[target_idx] -> 이진 코드 리스트
                
                # word2vec 구현에서 target_idx가 OOV일 경우 대비 코드가 필요할 수 있으나,
                # token_indices는 이미 min_count를 통과한 단어로 구성되었다고 가정합니다.
                
                path = self.path_table[target_idx]
                code = self.code_table[target_idx]

                # 5. 학습 쌍 Yield
                # center_idx: 중심 단어 인덱스
                # path: 타겟 단어의 Huffman Tree 경로 (노드 인덱스 리스트)
                # code: 경로를 따라가며 얻는 이진 코드 (0 또는 1 리스트)
                yield torch.tensor(center_idx, dtype=torch.long), path, code
                
            current_idx += 1 # 중심 단어 인덱스 증가
    
def collate_fn_hs(batch):
    centers, paths, codes = zip(*batch)
    
    max_len = max(len(p) for p in paths)
    batch_size = len(centers)
    
    padded_paths = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_codes = torch.zeros(batch_size, max_len, dtype=torch.float)
    masks = torch.zeros(batch_size, max_len, dtype=torch.float)
    
    for i, (p, c) in enumerate(zip(paths, codes)):
        l = len(p)
        padded_paths[i, :l] = torch.tensor(p)
        padded_codes[i, :l] = torch.tensor(c)
        masks[i, :l] = 1.0
    
    return torch.tensor(centers), padded_paths, padded_codes, masks

def get_dataloader(file_path, config, word2idx, word_freq, mode="ns", path_table=None, code_table=None):
    
    if mode == "ns":
        dataset = SkipGramNSIterableDataset(
            file_path=file_path, # 👈 파일 경로 전달
            word2idx=word2idx, 
            word_freq=word_freq, # 👈 빈도수 전달,
            device="cuda",
            neg_sample_size=config.get("neg_sample_size", 5),
            window_size=config["window_size"]
        )
        collate_fn = None 
        
    elif mode == "hs":
        if path_table is None or code_table is None:
            raise ValueError("path_table과 code_table이 없어요")
        dataset = SkipGramHSIterableDataset(
            file_path=file_path,
            word2idx=word2idx,
            word_freq=word_freq,
            path_table=path_table,
            code_table=code_table,
            window_size=config["window_size"]
        )
        collate_fn = collate_fn_hs
    else:
        raise ValueError("mode must be 'ns' or 'hs'")

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False, 
        num_workers=config.get("num_workers", 16), 
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader