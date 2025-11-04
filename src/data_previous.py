# src/data.py
from collections import Counter
import random
import csv
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
import pickle
from typing import Generator, List

import numpy as np

import xml.etree.ElementTree as ET

def subsample_text(text, t=1e-4): 
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

# def load_tokenized_corpus_from_file(path=TOKENIZED_SAVE_PATH) -> list:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Tokenized corpus file not found: {path}.")
#     print(f"Loading tokenized corpus from saved file: {path}...")
    
#     try:
#         with open(path, 'rb') as f:
#             tokens = pickle.load(f)
        
#         if not isinstance(tokens, list):
#             raise TypeError("Loaded object is not a list. Check the data format in the .pkl file.")

#         print(f"Tokenized corpus successfully loaded. Total tokens: {len(tokens)}")
#         return tokens
        
#     except Exception as e:
#         print(f"Error loading pickle file: {e}")
#         raise
TOKENIZED_TRAIN_PATH = "data/pretrain/tokenized_corpus.txt"
TOKEN_INDICES_PATH = "data/pretrain/token_indices.npy"

def word_stream_generator(file_path) -> Generator[List[str], None, None]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tokenized corpus file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for token in line.strip().split():
                yield token
                
def build_vocab_stream(file_path, min_count=10):
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

# def build_vocab(text, min_count=5):

#     tokens = text if isinstance(text, list) else text.lower().split()
#     vocab_counter = Counter(tokens)
#     vocab = {word: count for word, count in vocab_counter.items() if count >= min_count}

#     word2idx = {word: i for i, word in enumerate(vocab.keys())}
#     idx2word = {i: word for word, i in word2idx.items()}
    
#     print(f"📚 Built vocab: {len(vocab)} words (min_count={min_count})")
    
#     return vocab, word2idx, idx2word
class SkipGramNSIterableDataset(IterableDataset):
    def __init__(self, file_path, word2idx, word_freq, neg_sample_size=5, window_size=2, subsample_t=1e-4):
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
        self.sample_probs = self.sample_probs.to("cuda")
        
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
        
        freqs_list = [word_freq.get(self.idx2word.get(i), 0) for i in range(self.vocab_size)] 
        self.freqs_for_neg = torch.tensor(freqs_list, dtype=torch.float)
        self.freqs_for_neg[self.freqs_for_neg == 0] = 1e-30 # 0 방지
        self.sample_probs = self.freqs_for_neg.pow(0.75) / self.freqs_for_neg.pow(0.75).sum()
        
        # Subsampling 확률 계산
        self.total_count = sum(word_freq.values())
        self.freqs = {word: count / self.total_count for word, count in word_freq.items()}
        # 윈도우 크기 + 1 만큼 버퍼를 유지하여 context window를 만듭니다.
        # 실제 Word2Vec 학습은 토큰 인덱스 단위로 이루어집니다. 
        # 토큰 인덱스로 변환된 시퀀스를 사용하여 버퍼를 관리합니다.

    # def _line_to_index_stream(self):
    #     """파일을 한 줄씩 읽어 인덱스로 변환하고, 유효한 단어만 남기는 제너레이터."""
    #     with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
    #         for line in f:
    #             indices = [self.word2idx[token] for token in line.strip().split() if token in self.word2idx]
    #             yield indices

    # def __iter__(self):
    #     """학습 쌍 (center, context, neg_samples)을 yield"""
        
    #     # Word2Vec 학습을 위한 Sliding Window 버퍼
    #     buffer = [] 
        
    #     # 라인(문맥 단위) 인덱스 스트림을 가져옵니다.
    #     index_stream = self._line_to_index_stream()

    #     for indices in index_stream:
    #         # 새 문맥을 버퍼에 추가합니다.
    #         buffer.extend(indices)
            
    #         # 버퍼가 window_size를 초과하는 동안 반복
    #         while len(buffer) > 0:
    #             # 1. 중심 단어 설정 및 서브샘플링
    #             center_idx = buffer.pop(0) # 가장 오래된 토큰을 중심 단어로
    #             center_token = self.idx2word[center_idx]
                
    #             # 서브샘플링 확률
    #             f = self.freqs.get(center_token, 0)
    #             p_drop = 1 - ((self.subsample_t / f) ** 0.5) if f > 0 else 1
    #             if random.random() < p_drop: 
    #                 continue # 드롭

    #             # 2. 가변 윈도우 설정
    #             actual_window = random.randint(1, self.window_size)
                
    #             # 3. 주변 단어(Context) 선택
    #             # 현재 버퍼(중심단어 이후)와 버퍼 앞(중심단어 이전, 즉 pop된 토큰)의 토큰을 모두 고려해야 하지만,
    #             # IterableDataset의 스트리밍 특성상 이전 토큰은 재구성이 어렵습니다.
    #             # 여기서는 *현재 버퍼의 토큰*을 주변 단어로 사용합니다.
                
    #             context_indices = buffer[:actual_window] # 오른쪽 문맥만 고려 (간소화)
                
    #             for context_idx in context_indices:
                    
    #                 # 4. 네거티브 샘플링
    #                 neg_samples = torch.multinomial(
    #                     self.sample_probs,
    #                     self.neg_sample_size,
    #                     replacement=True
    #                 )
                    
    #                 # 5. 학습 쌍 Yield
    #                 yield torch.tensor(center_idx, dtype=torch.long), \
    #                       torch.tensor(context_idx, dtype=torch.long), \
    #                       neg_samples   
    
    def __iter__(self):
        """학습 쌍 (center, context, neg_samples)을 yield"""
        
        # 🟢 [수정] 워커별 데이터 분할 (mmap 배열 인덱스를 사용하여 병렬 처리)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx = 0
            end_idx = self.total_tokens
        else:
            per_worker = self.total_tokens // worker_info.num_workers
            start_idx = worker_info.id * per_worker
            end_idx = start_idx + per_worker
            if worker_info.id == worker_info.num_workers - 1:
                 end_idx = self.total_tokens # 마지막 워커는 나머지를 모두 처리
                 
        current_idx = start_idx
        
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
            left_context_start = max(start_idx, current_idx - actual_window)
            right_context_end = min(end_idx, current_idx + actual_window + 1)
            
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

# def get_dataloader(text, config, word2idx, vocab=None, mode="ns"):
#     tokens = text if isinstance(text, list) else text.lower().split()

#     if mode == "ns":
#         dataset = SkipGramNSDataset(
#             tokens=tokens, 
#             word2idx=word2idx, 
#             vocab=vocab, 
#             neg_sample_size=config.get("neg_sample_size", 5),
#             window_size=config["window_size"]
#         )
#     elif mode == "hs":
#         dataset = SkipGramHSDataset(
#             tokens, 
#             word2idx,
#             window_size=config["window_size"]
#         )
#     else:
#         raise ValueError("mode must be 'ns' or 'hs'")

#     dataloader = DataLoader(
#         dataset,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         num_workers=config.get("num_workers", 4), 
#         pin_memory=True
#     )
#     return dataloader

def get_dataloader(file_path, config, word2idx, word_freq, mode="ns", path_table=None, code_table=None):
    
    if mode == "ns":
        dataset = SkipGramNSIterableDataset(
            file_path=TOKEN_INDICES_PATH, # 👈 파일 경로 전달
            word2idx=word2idx, 
            word_freq=word_freq, # 👈 빈도수 전달
            neg_sample_size=config.get("neg_sample_size", 5),
            window_size=config["window_size"]
        )
        
    elif mode == "hs":
        # HS IterableDataset 구현 시 여기에 추가
        raise NotImplementedError("HS IterableDataset must be implemented for streaming.")
        
    else:
        raise ValueError("mode must be 'ns' or 'hs'")

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        # IterableDataset은 데이터 순서를 Dataset 내부에서 처리하므로 shuffle=False
        shuffle=False, 
        # OOM 방지를 위해 num_workers는 0으로 권장, 특히 파일 IO가 많을 때
        num_workers=config.get("num_workers", 32), 
        pin_memory=True
    )
    return dataloader