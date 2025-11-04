# src/eval.py
import torch
import argparse
import yaml
import numpy as np
import csv
import os
from scipy.stats import spearmanr
from model import SkipGram
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Word2Vec Evaluation (SkipGram NS/HS)")
parser.add_argument("config", type=str, help="Path to YAML config")
parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pth)")
parser.add_argument("wordsim_csv", type=str, help="Path to WordSim-353 CSV file")
parser.add_argument("simlex_txt", type=str, help="Path to SimLex-999 text file")
parser.add_argument("google_analogy", type=str, help="Path to Google analogy questions file")
parser.add_argument("--save_csv", type=str, default=None, help="Optional: save evaluation results to CSV")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"\n🧠 Using device: {device}")

if config["training_mode"] == "hs":
  with open("runs/vocab_hs.pkl", "rb") as f:
      vocab_data = pickle.load(f)
else:
  with open("data/pretrain/vocab_data_3.pkl", "rb") as f:
      vocab_data = pickle.load(f)
vocab, word2idx, idx2word = vocab_data["vocab"], vocab_data["word2idx"], vocab_data["idx2word"]

vocab_size = len(vocab)
embedding_dim = config["embedding_dim"]

print(f"📚 Loaded vocab: {vocab_size} words")

mode = config.get("training_mode", "ns").lower()  
model = SkipGram(vocab_size, embedding_dim).to(device)

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"✅ Loaded model ({mode.upper()}) from: {args.checkpoint}")


print("\n⚡ Caching all word embeddings...")
with torch.no_grad():
    in_weights = model.in_embeddings.weight.detach().to(device)
    
    if mode == 'ns' and model.out_embeddings.weight.size(0) == vocab_size:
        print("💡 Combining In and Out Embeddings (Average)...")
        out_weights = model.out_embeddings.weight.detach().to(device)
        embedding_matrix = (in_weights + out_weights) / 2
    else:
        print("⚠️ Using only In Embeddings (V)...")
        embedding_matrix = in_weights
    
    embedding_matrix = embedding_matrix.half()
        
print("Embedding matrix cached.")


def get_embedding(word):
    """단어 임베딩 벡터 반환"""
    if word not in word2idx:
        return None
    return embedding_matrix[word2idx[word]]


def cosine_similarity(v1, v2):
    """코사인 유사도 계산"""
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))


@torch.no_grad()
def analogy(a, b, c, topk=1):
    """a:b = c:? (GPU 병렬 버전)"""
    e_a, e_b, e_c = get_embedding(a), get_embedding(b), get_embedding(c)
    if e_a is None or e_b is None or e_c is None:
        return []
    target_vec = e_b - e_a + e_c
    target_vec = target_vec.half()
    sims = torch.nn.functional.cosine_similarity(embedding_matrix, target_vec.unsqueeze(0), dim=1)

    for w in [a, b, c]:
        if w in word2idx:
            sims[word2idx[w]] = -float("inf")

    topk_idx = torch.topk(sims, topk).indices
    return [(idx2word[i.item()], sims[i].item()) for i in topk_idx]

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
            if line.startswith(":") or len(line) == 0:
                continue
            words = line.lower().split()
            if len(words) == 4:
                analogies.append(tuple(words))
    return analogies

@torch.no_grad()
def evaluate_similarity(pairs):
    preds, golds = [], []
    for w1, w2, sim in tqdm(pairs, desc="🔹 Word Similarity", ncols=100):
        e1, e2 = get_embedding(w1), get_embedding(w2)
        if e1 is None or e2 is None:
            continue
        preds.append(cosine_similarity(e1, e2).cpu().item())
        golds.append(sim)
    corr, _ = spearmanr(preds, golds)
    return corr


@torch.no_grad()
def evaluate_analogy(analogies):
    total, correct = 0, 0
    for a, b, c, d_true in tqdm(analogies, desc="🔸 Analogy", ncols=100):
        preds = analogy(a, b, c, topk=1)
        if not preds:
            continue
        total += 1
        if preds[0][0] == d_true:
            correct += 1
    return correct / total if total > 0 else 0.0

print("\n Starting evaluation...\n")
results = {}

# WordSim-353
ws_pairs = load_wordsim353(args.wordsim_csv)
ws_corr = evaluate_similarity(ws_pairs)
results["WordSim-353"] = ws_corr
print(f"📊 WordSim-353 Spearman: {ws_corr:.4f}")

# SimLex-999
simlex_pairs = load_simlex999(args.simlex_txt)
simlex_corr = evaluate_similarity(simlex_pairs)
results["SimLex-999"] = simlex_corr
print(f"📘 SimLex-999 Spearman: {simlex_corr:.4f}")

# Google Analogy
analogy_pairs = load_google_analogy(args.google_analogy)
ga_acc = evaluate_analogy(analogy_pairs)
results["Google Analogy"] = ga_acc
print(f"👑 Google Analogy Accuracy: {ga_acc:.4f}")

print("\n==============================")
print("Evaluation Summary")
print("==============================")
for name, val in results.items():
    metric = "Spearman" if "Sim" in name else "Accuracy"
    print(f"{name:<15}: {val:.4f} ({metric})")

# Optional CSV 저장
if args.save_csv:
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Metric", "Score"])
        for name, val in results.items():
            metric = "Spearman" if "Sim" in name else "Accuracy"
            writer.writerow([name, metric, f"{val:.4f}"])
    print(f"\nResults saved to: {args.save_csv}")
