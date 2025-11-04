import torch
import torch.nn as nn
import torch.optim as optim
import yaml, argparse, random, os, pickle
import numpy as np
from tqdm import tqdm
import time
import csv # [추가] CSV 파일 저장을 위해 csv 모듈 import

from data import build_vocab_stream, get_dataloader, TOKENIZED_TRAIN_PATH
from model import SkipGram
from huffman_tree import HuffmanTree  

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
TOKEN_INDICES_PATH = "data/pretrain/token_indices.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/ns_window-2_epoch-5.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config["seed"])

# [수정] 학습 시작 시간 기록
start_time = time.time() 

# text = load_tokenized_corpus_from_file()

# vocab, word2idx, idx2word = build_vocab(text)
# text = [token for token in text if token in word2idx]
# text = subsample_text(text, t=1e-4)

vocab, word2idx, idx2word, word_freq = build_vocab_stream(
    TOKENIZED_TRAIN_PATH, 
    min_count=config.get("min_count", 5) 
)
vocab_size = len(vocab)
# dataloader = get_dataloader(text, config, word2idx, vocab, mode=config["training_mode"])

dataloader = get_dataloader(
    TOKEN_INDICES_PATH, 
    config, 
    word2idx, 
    word_freq, # 👈 빈도수 전달
    mode=config["training_mode"]
)

embedding_dim = config["embedding_dim"]
model = SkipGram(vocab_size, embedding_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device.type == 'cuda':
    torch.cuda.reset_max_memory_allocated()
    print(f"🔥 Device: {device}, Initial GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

optimizer = optim.Adam(model.parameters(), lr=config["lr"])

if config["training_mode"] == "hs":
    print("🧩 Building Huffman Tree...")
    word_freq = {w: vocab[w] for w in vocab}
    huffman_tree = HuffmanTree(word_freq)
    path_table = {}
    code_table = {}
    for word, idx in word2idx.items():
        path_table[idx] = huffman_tree.get_path(idx)
        code_table[idx] = huffman_tree.get_code(idx)

num_epochs = config["epochs"]
checkpoint_dir = f"runs/checkpoints_{config['training_mode']}"
os.makedirs(checkpoint_dir, exist_ok=True)
try:
    total_batches = len(dataloader) 
except TypeError:
    print("⚠️ Warning: IterableDataset has no definite length. Using estimated steps.")
    # 실제 토큰 수 기반으로 대략적인 배치를 계산해야 함.
    total_batches = 500000
    
total_steps = config["epochs"] * total_batches
current_step = 0
print(f"🚀 Training SkipGram ({config['training_mode'].upper()}) mode...")

# [추가] 에폭별 학습 지표를 저장할 리스트
metrics_log = [] 

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    total_loss = 0
    model.train()
    
    # estimated_steps = 800  
    progress_bar = tqdm(
        dataloader,
        total=total_batches,
        desc=f"Epoch {epoch}/{num_epochs}",
        dynamic_ncols=True
    )
    
    batch_count_in_epoch = 0

    # for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
    for batch in progress_bar:
        current_step += 1
        batch_count_in_epoch += 1
        
        progress = current_step / total_steps 
        new_lr = config["lr"] * (1 - progress) 
        new_lr = max(0.0001, new_lr) 
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        optimizer.zero_grad()

        if config["training_mode"] == "ns":
            center, pos_context, neg_samples = batch
            loss = model.forward_ns(center.to(device), pos_context.to(device), neg_samples.to(device))

        elif config["training_mode"] == "hs":
            # center_idx, target_idx = batch
            # path_idx = [path_table[t.item()] for t in target_idx]
            # code_tensor = [code_table[t.item()] for t in target_idx]
            # loss = model.forward_hs(center_idx.to(device), path_idx, code_tensor)
            raise NotImplementedError("HS streaming mode is not yet fully implemented.")


        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        avg_loss = total_loss / batch_count_in_epoch
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "lr": f"{new_lr:.6f}"
        })

    progress_bar.close()


    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    # avg_loss = total_loss / len(dataloader)
    avg_loss = total_loss / batch_count_in_epoch
    
    max_memory = 0
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        torch.cuda.reset_max_memory_allocated() # 다음 에폭 측정을 위해 리셋
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f}s, Max GPU Memory: {max_memory:.2f} GB")
    else:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f}s")
    
    
    # [추가] 에폭별 지표를 로그 리스트에 저장
    metrics_log.append({
        "epoch": epoch,
        "loss": avg_loss,
        "duration_seconds": epoch_duration,
        "max_gpu_memory_gb": max_memory,
    })

    ckpt_path = os.path.join(checkpoint_dir, f"{config['training_mode']}_window-{config['window_size']}_epoch-{epoch}.pth")
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "config": config,
        "epoch_duration": epoch_duration,
        "max_gpu_memory": max_memory if device.type == 'cuda' else None,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

# [수정] 전체 학습 시간 기록 및 CSV 저장
end_time = time.time()
total_training_duration = end_time - start_time
print(f"\n✅ Total Training Time: {total_training_duration:.2f} seconds ({total_training_duration / 3600:.2f} GPU-hours)")

# ----------------- CSV 저장 로직 -----------------
# [추가] CSV 파일 이름 설정 및 저장 경로 생성
csv_filename = f"metrics_{config['training_mode']}_window-{config['window_size']}_seed-{config['seed']}.csv"
csv_path = os.path.join("runs/results", csv_filename)

if metrics_log:
    # 헤더 정의
    fieldnames = ["epoch", "loss", "duration_seconds", "max_gpu_memory_gb"]
    
    # runs 폴더가 없으면 생성
    os.makedirs("runs/results", exist_ok=True)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(metrics_log)

    print(f"📊 Metrics saved to {csv_path}")
else:
    print("⚠️ Warning: Metrics log is empty. CSV file not created.")
# -------------------------------------------------

vocab_data = {"vocab": vocab, "word2idx": word2idx, "idx2word": idx2word}
training_mode = config.get("training_mode", "default")
vocab_filename = f"vocab_{training_mode}.pkl"
vocab_path = os.path.join("runs", vocab_filename)

with open(vocab_path, "wb") as f:
    pickle.dump(vocab_data, f)

print("Training finished successfully!")