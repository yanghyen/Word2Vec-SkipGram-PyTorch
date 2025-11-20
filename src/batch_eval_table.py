#!/usr/bin/env python3
"""
배치 평가 테이블 생성 스크립트: runs/checkpoints_ns와 runs/checkpoints_hs의 모든 .pth 파일들을 평가하여 
파일명을 컬럼으로 하는 하나의 CSV 테이블로 저장합니다.

사용법:
    python src/batch_eval_table.py [--output results/batch_evaluation_table.csv]
    python src/batch_eval_table.py --checkpoint_dir runs/checkpoints_ns [--output results/batch_evaluation_table.csv]
"""

import os
import glob
import torch
import yaml
import numpy as np
import pandas as pd
import pickle
import argparse
import re
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
import sys

from model import SkipGram

def parse_checkpoint_name(checkpoint_path):
    """
    체크포인트 파일명에서 설정 정보를 추출합니다.
    
    예시: ns_window-5_sub-False_seed-123.pth -> ns, window=5, subsample=False, seed=123
    """
    filename = Path(checkpoint_path).stem
    
    info = {
        'mode': 'ns',
        'window': 5,
        'subsample': True,
        'seed': 42
    }
    
    # 모드 추출
    if filename.startswith('hs'):
        info['mode'] = 'hs'
    elif filename.startswith('ns'):
        info['mode'] = 'ns'
    
    # window 크기 추출
    window_match = re.search(r'window-(\d+)', filename)
    if window_match:
        info['window'] = int(window_match.group(1))
    
    # subsample 설정 추출
    if 'sub-False' in filename:
        info['subsample'] = False
    elif 'sub-True' in filename:
        info['subsample'] = True
    
    # seed 추출
    seed_match = re.search(r'seed-(\d+)', filename)
    if seed_match:
        info['seed'] = int(seed_match.group(1))
    
    return info

def find_matching_config(checkpoint_info, configs_dir="configs"):
    """
    체크포인트 정보에 맞는 config 파일을 찾습니다.
    """
    mode = checkpoint_info['mode']
    window = checkpoint_info['window']
    subsample = 'on' if checkpoint_info['subsample'] else 'off'
    seed = checkpoint_info['seed']
    
    # 가능한 config 파일명 패턴들
    patterns = [
        f"{mode}_window-{window}_subsample-{subsample}_seed-{seed}.yaml",
        f"{mode}_window-{window}_subsample-{subsample}_seed-42.yaml",  # fallback
    ]
    
    for pattern in patterns:
        config_path = os.path.join(configs_dir, pattern)
        if os.path.exists(config_path):
            return config_path
    
    return None

def load_wordsim353(csv_path):
    """WordSim-353 데이터셋 로드"""
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    w1, w2, score = parts[0], parts[1], float(parts[2])
                    pairs.append((w1, w2, score))
                except:
                    continue
    return pairs

def load_simlex999(txt_path):
    """SimLex-999 데이터셋 로드"""
    pairs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        next(f)  # 헤더 스킵
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    w1, w2, score = parts[0], parts[1], float(parts[3])
                    pairs.append((w1, w2, score))
                except:
                    continue
    return pairs

def load_google_analogy(txt_path):
    """Google Analogy 데이터셋 로드"""
    analogies = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(':'):
                continue
            parts = line.strip().split()
            if len(parts) == 4:
                analogies.append(tuple(parts))
    return analogies

def cosine_similarity_gpu(embeddings_tensor, v1_idx, v2_idx):
    """GPU를 사용한 코사인 유사도 계산"""
    v1 = embeddings_tensor[v1_idx]
    v2 = embeddings_tensor[v2_idx]
    
    dot_product = torch.dot(v1, v2)
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return (dot_product / (norm_v1 * norm_v2)).item()

def cosine_similarity(v1, v2):
    """CPU용 코사인 유사도 계산 (백업용)"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)

def analogy_gpu(embeddings_tensor, word2idx, a, b, c, topk=1, device='cuda'):
    """GPU를 사용한 단어 유추 수행: a is to b as c is to ?"""
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return []
    
    # 벡터 연산: king - man + woman = queen
    vec_result = embeddings_tensor[word2idx[b]] - embeddings_tensor[word2idx[a]] + embeddings_tensor[word2idx[c]]
    
    # 입력 단어들의 인덱스
    exclude_indices = {word2idx[a], word2idx[b], word2idx[c]}
    
    # 모든 임베딩과의 코사인 유사도를 배치로 계산
    # 정규화
    vec_result_norm = vec_result / torch.norm(vec_result)
    embeddings_norm = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)
    
    # 배치 코사인 유사도 계산
    similarities = torch.matmul(embeddings_norm, vec_result_norm)
    
    # 입력 단어들 제외
    for idx in exclude_indices:
        similarities[idx] = -float('inf')
    
    # 상위 topk 개 찾기
    _, top_indices = torch.topk(similarities, topk)
    
    # 결과 변환
    idx2word = {idx: word for word, idx in word2idx.items()}
    results = []
    for i in range(topk):
        idx = top_indices[i].item()
        sim = similarities[idx].item()
        if idx in idx2word:
            results.append((idx2word[idx], sim))
    
    return results

def analogy(embeddings, word2idx, a, b, c, topk=1):
    """CPU용 단어 유추 수행 (백업용)"""
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return []
    
    # 벡터 연산: king - man + woman = queen
    vec_result = embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]
    
    # 모든 단어와의 유사도 계산
    similarities = []
    for word, idx in word2idx.items():
        if word in [a, b, c]:  # 입력 단어들 제외
            continue
        sim = cosine_similarity(vec_result, embeddings[idx])
        similarities.append((word, sim))
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:topk]

def evaluate_similarity_gpu(embeddings_tensor, word2idx, pairs, device='cuda'):
    """GPU를 사용한 유사도 평가"""
    preds, golds = [], []
    
    for w1, w2, gold in pairs:
        if w1 in word2idx and w2 in word2idx:
            sim = cosine_similarity_gpu(embeddings_tensor, word2idx[w1], word2idx[w2])
            preds.append(sim)
            golds.append(gold)
    
    if len(preds) == 0:
        return 0.0
    
    corr, _ = spearmanr(preds, golds)
    return corr if not np.isnan(corr) else 0.0

def evaluate_similarity(embeddings, word2idx, pairs):
    """CPU용 유사도 평가 (백업용)"""
    preds, golds = [], []
    
    for w1, w2, gold in pairs:
        if w1 in word2idx and w2 in word2idx:
            sim = cosine_similarity(embeddings[word2idx[w1]], embeddings[word2idx[w2]])
            preds.append(sim)
            golds.append(gold)
    
    if len(preds) == 0:
        return 0.0
    
    corr, _ = spearmanr(preds, golds)
    return corr if not np.isnan(corr) else 0.0

def evaluate_analogy_gpu(embeddings_tensor, word2idx, analogies, device='cuda'):
    """GPU를 사용한 유추 평가"""
    total, correct = 0, 0
    for a, b, c, d_true in tqdm(analogies, desc="🔸 GPU Analogy", ncols=100):
        preds = analogy_gpu(embeddings_tensor, word2idx, a, b, c, topk=1, device=device)
        if not preds:
            continue
        total += 1
        if preds[0][0] == d_true:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def evaluate_analogy(embeddings, word2idx, analogies):
    """CPU용 유추 평가 (백업용)"""
    total, correct = 0, 0
    
    for a, b, c, d_true in analogies:
        preds = analogy(embeddings, word2idx, a, b, c, topk=1)
        if not preds:
            continue
        total += 1
        if preds[0][0] == d_true:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def evaluate_single_model(checkpoint_path, config_path):
    """단일 모델 평가"""
    print(f"📄 평가 중: {Path(checkpoint_path).name}")
    print(f"   Config: {config_path}")
    
    # Config 로드
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"   ✅ Config 로드 완료")
    except Exception as e:
        print(f"   ❌ Config 로드 실패: {e}")
        raise
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Vocab 로드
    if config["training_mode"] == "hs":
        vocab_path = "runs/vocab_hs.pkl"
    else:
        vocab_path = "data/pretrain/vocab_data_3.pkl"
    
    print(f"   📚 Vocab 로딩: {vocab_path}")
    try:
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)
        print(f"   ✅ Vocab 로드 완료")
    except Exception as e:
        print(f"   ❌ Vocab 로드 실패: {e}")
        raise
    
    vocab, word2idx, idx2word = vocab_data["vocab"], vocab_data["word2idx"], vocab_data["idx2word"]
    
    # 모델 로드
    vocab_size = len(vocab)
    embedding_dim = config["embedding_dim"]
    mode = config.get("training_mode", "ns").lower()
    
    print(f"   🧠 모델 생성: vocab_size={vocab_size}, embedding_dim={embedding_dim}, mode={mode}")
    try:
        model = SkipGram(vocab_size, embedding_dim, mode=mode).to(device)
        print(f"   ✅ 모델 생성 완료")
        
        print(f"   📦 체크포인트 로딩: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"   ✅ 모델 로드 완료")
    except Exception as e:
        print(f"   ❌ 모델 로드 실패: {e}")
        raise
    
    # 임베딩 추출 (GPU 버전과 CPU 버전 모두 준비)
    print(f"   🔢 임베딩 추출 중...")
    try:
        with torch.no_grad():
            embeddings_tensor = model.in_embeddings.weight.detach()  # GPU에 유지
            embeddings = embeddings_tensor.cpu().numpy()  # CPU 백업용
        print(f"   ✅ 임베딩 추출 완료: {embeddings.shape}")
        print(f"   🚀 GPU 가속 사용: {device}")
    except Exception as e:
        print(f"   ❌ 임베딩 추출 실패: {e}")
        raise
    
    # 데이터셋 로드
    print(f"   📊 데이터셋 로딩 중...")
    try:
        wordsim_pairs = load_wordsim353("data/word_similarity/combined.csv")
        print(f"   ✅ WordSim-353 로드: {len(wordsim_pairs)}개 쌍")
        
        simlex_pairs = load_simlex999("data/word_similarity/SimLex-999/SimLex-999.txt")
        print(f"   ✅ SimLex-999 로드: {len(simlex_pairs)}개 쌍")
        
        analogy_pairs = load_google_analogy("data/word_similarity/word2vec/trunk/questions-words.txt")
        print(f"   ✅ Google Analogy 로드: {len(analogy_pairs)}개 쌍")
    except Exception as e:
        print(f"   ❌ 데이터셋 로드 실패: {e}")
        raise
    
    # 평가 수행 (GPU 가속 사용)
    print(f"   🎯 평가 시작...")
    results = {}
    
    try:
        print(f"   📊 WordSim-353 평가 중... (GPU 가속)")
        results["WordSim-353"] = evaluate_similarity_gpu(embeddings_tensor, word2idx, wordsim_pairs, device)
        print(f"   ✅ WordSim-353: {results['WordSim-353']:.4f}")
        
        print(f"   📘 SimLex-999 평가 중... (GPU 가속)")
        results["SimLex-999"] = evaluate_similarity_gpu(embeddings_tensor, word2idx, simlex_pairs, device)
        print(f"   ✅ SimLex-999: {results['SimLex-999']:.4f}")
        
        print(f"   👑 Google Analogy 평가 중... (GPU 가속으로 빨라집니다!)")
        results["Google Analogy"] = evaluate_analogy_gpu(embeddings_tensor, word2idx, analogy_pairs, device)
        print(f"   ✅ Google Analogy: {results['Google Analogy']:.4f}")
        
    except Exception as e:
        print(f"   ⚠️ GPU 평가 실패, CPU로 대체: {e}")
        # GPU 실패시 CPU 백업
        print(f"   📊 WordSim-353 평가 중... (CPU)")
        results["WordSim-353"] = evaluate_similarity(embeddings, word2idx, wordsim_pairs)
        print(f"   ✅ WordSim-353: {results['WordSim-353']:.4f}")
        
        print(f"   📘 SimLex-999 평가 중... (CPU)")
        results["SimLex-999"] = evaluate_similarity(embeddings, word2idx, simlex_pairs)
        print(f"   ✅ SimLex-999: {results['SimLex-999']:.4f}")
        
        print(f"   👑 Google Analogy 평가 중... (CPU - 시간이 오래 걸립니다)")
        results["Google Analogy"] = evaluate_analogy(embeddings, word2idx, analogy_pairs)
        print(f"   ✅ Google Analogy: {results['Google Analogy']:.4f}")
    
    print(f"   🎉 평가 완료!")
    return results

def main():
    parser = argparse.ArgumentParser(description="배치 평가 테이블 생성")
    parser.add_argument("--checkpoint_dir", default=None, help="체크포인트 파일들이 있는 디렉토리 (지정하지 않으면 runs/checkpoints_ns와 runs/checkpoints_hs에서 자동 검색)")
    parser.add_argument("--configs_dir", default="configs", help="config 파일들이 있는 디렉토리")
    parser.add_argument("--output", default="results/batch_evaluation_table.csv", help="출력 CSV 파일 경로")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # .pth 파일들 찾기
    if args.checkpoint_dir:
        # 특정 디렉토리에서 찾기
        checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, "*.pth"))
        checkpoint_files.sort()
        print(f"📁 지정된 디렉토리에서 {len(checkpoint_files)}개의 체크포인트 파일을 찾았습니다: {args.checkpoint_dir}")
    else:
        # runs/checkpoints_ns와 runs/checkpoints_hs에서 자동으로 찾기
        checkpoint_files = []
        checkpoint_dirs = ["runs/checkpoints_ns", "runs/checkpoints_hs"]
        for ckpt_dir in checkpoint_dirs:
            if os.path.exists(ckpt_dir):
                files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
                checkpoint_files.extend(files)
                if files:
                    print(f"📁 {ckpt_dir}에서 {len(files)}개의 체크포인트 파일을 찾았습니다.")
        checkpoint_files.sort()
        print(f"📁 총 {len(checkpoint_files)}개의 체크포인트 파일을 찾았습니다.")
    
    # 각 모델 평가
    all_results = {}
    
    for checkpoint_path in tqdm(checkpoint_files, desc="모델 평가"):
        try:
            # 체크포인트 정보 추출
            checkpoint_info = parse_checkpoint_name(checkpoint_path)
            
            # 매칭되는 config 파일 찾기
            config_path = find_matching_config(checkpoint_info, args.configs_dir)
            
            if not config_path:
                print(f"❌ Config 파일을 찾을 수 없습니다: {checkpoint_path}")
                continue
            
            # 모델 평가
            results = evaluate_single_model(checkpoint_path, config_path)
            
            # 파일명을 키로 사용 (확장자 제거)
            model_name = Path(checkpoint_path).stem
            all_results[model_name] = results
            
        except Exception as e:
            print(f"❌ 평가 실패: {checkpoint_path}")
            print(f"   에러: {e}")
            continue
    
    # 결과를 DataFrame으로 변환
    if not all_results:
        print("❌ 평가된 모델이 없습니다.")
        return
    
    # 데이터셋을 행으로, 모델을 열로 하는 테이블 생성
    datasets = ["WordSim-353", "SimLex-999", "Google Analogy"]
    
    table_data = {}
    for dataset in datasets:
        table_data[dataset] = {}
        for model_name, results in all_results.items():
            table_data[dataset][model_name] = results.get(dataset, 0.0)
    
    # DataFrame 생성
    df = pd.DataFrame(table_data).T  # 전치하여 데이터셋이 행이 되도록
    
    # CSV 저장
    df.to_csv(args.output)
    
    print(f"\n✅ 배치 평가 완료!")
    print(f"📊 평가된 모델 수: {len(all_results)}")
    print(f"💾 결과 저장됨: {args.output}")
    
    # 결과 미리보기
    print(f"\n📋 결과 미리보기:")
    print(df.round(4))

if __name__ == "__main__":
    main()
