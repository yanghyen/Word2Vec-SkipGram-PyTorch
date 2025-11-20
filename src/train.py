# 대규모 코퍼스 돌리고 loss 증가했던 코드 (251029)

import torch
import math
import torch.nn as nn
import torch.optim as optim
import yaml, argparse, random, os, pickle
import numpy as np
from tqdm import tqdm
import time
import csv 

from data import get_dataloader 
from model import SkipGram
from huffman_tree import HuffmanTree 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
TOKEN_INDICES_PATH = "data/pretrain/token_indices_3.npy"
VOCAB_FILENAME = "vocab_data_3.pkl"
VOCAB_PATH = os.path.join("data/pretrain", VOCAB_FILENAME)

# ----------------- Arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/ns_window-5_subsampling-on_seed-42.yaml")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
args = parser.parse_args()

# ----------------- Load config -----------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# ----------------- Seed -----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config["seed"])

# ----------------- Load Vocab -----------------
try:
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found: {VOCAB_PATH}")
        
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
        
    vocab = vocab_data["vocab"]
    word2idx = vocab_data["word2idx"]
    idx2word = vocab_data["idx2word"]
    word_freq = vocab_data.get("word_freq", {w: vocab[w] for w in vocab}) 
    
    vocab_size = len(vocab)
    print(f"✅ Loaded Vocab successfully from {VOCAB_PATH}. Size: {vocab_size:,}")

except FileNotFoundError as e:
    print(f"❌ 오류: {e}")
    exit()
except Exception as e:
    print(f"❌ Vocab 로드 중 치명적 오류: {e}")
    exit()

if config["training_mode"] == "hs":
    print("🧩 Building Huffman Tree...")
    huffman_tree = HuffmanTree(word_freq)
    
    vocab_size = len(word2idx)
    path_table = [[] for _ in range(vocab_size)]
    code_table = [[] for _ in range(vocab_size)]
    
    for idx in range(vocab_size):
        path_table[idx] = huffman_tree.get_path(idx)
        code_table[idx] = huffman_tree.get_code(idx)
    print("huffman tree 만들어졌고 HS table 완성 (list or list)")
else:
    path_table = None
    code_table = None
# ----------------- DataLoader -----------------
dataloader = get_dataloader(
    TOKEN_INDICES_PATH, 
    config, 
    word2idx, 
    word_freq,
    mode=config["training_mode"],
    path_table=path_table,
    code_table=code_table,
)

# vocab_size 검증 
print("🔍 Validating token indices...")
sample_indices = np.load(TOKEN_INDICES_PATH, mmap_mode='r')[:1000]  # 샘플링
if np.any(sample_indices >= vocab_size):
    raise ValueError(f"Invalid indices found! Max index: {sample_indices.max()}, vocab_size: {vocab_size}")
print("✅ Indices validation passed")

# ----------------- Model & Optimizer -----------------
embedding_dim = config["embedding_dim"]
model = SkipGram(vocab_size, embedding_dim, mode=config["training_mode"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 안전한 학습률
initial_lr = config.get("lr", 0.001)
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# ----------------- Huffman Tree (HS mode) -----------------
# ----------------- Checkpoint setup -----------------
checkpoint_dir = f"runs/checkpoints_{config['training_mode']}"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_every = config.get("checkpoint_every", 500000)

start_epoch = 1
current_step = 0

# ----------------- Resume from checkpoint -----------------
if args.resume is not None and os.path.exists(args.resume):
    print(f"🔄 Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    current_step = checkpoint.get("current_step", 0)
    print(f"✅ Resuming training from epoch {start_epoch}, step {current_step}")


# ----------------- Training Loop -----------------
num_epochs = config["epochs"]
metrics_log = []

try:
    total_batches = len(dataloader) 
except TypeError:
    print("⚠️ Warning: IterableDataset has no definite length. Using estimated steps (500k).")
    total_batches = 2400000
    
total_steps = num_epochs * total_batches

start_time = time.time()

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    total_loss = 0
    batch_count_in_epoch = 0
    model.train()
    
    progress_bar = tqdm(
        dataloader,
        total=total_batches,
        desc=f"Epoch {epoch}/{num_epochs}",
        dynamic_ncols=True
    )
    
    for batch in progress_bar:
        current_step += 1
        batch_count_in_epoch += 1
        
        # ----------------- Learning rate decay -----------------
        progress = current_step / total_steps
        min_lr = initial_lr * 0.1  # 최소 lr을 초기값의 10%로
        new_lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        optimizer.zero_grad()

        # ----------------- Forward -----------------
        if config["training_mode"] == "ns":
            center, pos_context, neg_samples = batch
            
            loss = model.forward_ns(center.to(device), pos_context.to(device), neg_samples.to(device))
        
        elif config["training_mode"] == "hs":
            center, paths, codes, masks = batch
            
            loss = model.forward_hs(
                center.to(device), 
                paths.to(device), 
                codes.to(device),
                masks.to(device)
            )
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss at step {current_step}, skipping")
            continue
        
        # ----------------- Backprop + Gradient Clipping -----------------
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / batch_count_in_epoch
        
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "lr": f"{new_lr:.6f}"
        })
        
        # ----------------- Step Checkpoint -----------------
        # if current_step % checkpoint_every == 0:
        #     ckpt_path = os.path.join(checkpoint_dir, f"{config['training_mode']}_window-{config['window_size']}_sub-{config['enable_subsampling']}_seed-{config['seed']}_step-{current_step}.pth")
        #     torch.save({
        #         "epoch": epoch,
        #         "current_step": current_step,
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #     }, ckpt_path)
        #     print(f"💾 Saved intermediate checkpoint at step {current_step}")

    progress_bar.close()
    
    epoch_duration = time.time() - epoch_start_time
    avg_loss = total_loss / batch_count_in_epoch
    
    max_memory = 0
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        torch.cuda.reset_max_memory_allocated()
    
    metrics_log.append({
        "epoch": epoch,
        "loss": avg_loss,
        "duration_seconds": epoch_duration,
        "max_gpu_memory_gb": max_memory,
    })
    
    # ----------------- Epoch Checkpoint -----------------
    ckpt_path = os.path.join(checkpoint_dir, f"{config['training_mode']}_window-{config['window_size']}_sub-{config['enable_subsampling']}_seed-{config['seed']}.pth")
    torch.save({
        "epoch": epoch,
        "current_step": current_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "epoch_duration": epoch_duration,
        "max_gpu_memory": max_memory if device.type == 'cuda' else None,
    }, ckpt_path)
    print(f"💾 Saved checkpoint for epoch {epoch}")

# ----------------- Metrics CSV 저장 -----------------
csv_filename = f"metrics_{config['training_mode']}_window-{config['window_size']}_sub-{config['enable_subsampling']}_seed-{config['seed']}.csv"
csv_path = os.path.join("runs/metrics", csv_filename)
os.makedirs("runs/metrics/", exist_ok=True)
if metrics_log:
    fieldnames = ["epoch", "loss", "duration_seconds", "max_gpu_memory_gb"]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_log)
    print(f"📊 Metrics saved to {csv_path}")

# ----------------- Final Vocab 저장 -----------------
vocab_data = {"vocab": vocab, "word2idx": word2idx, "idx2word": idx2word}
vocab_filename = f"vocab_{config['training_mode']}.pkl" 
vocab_path = os.path.join("runs", vocab_filename)
with open(vocab_path, "wb") as f:
    pickle.dump(vocab_data, f)
print(f"✅ Final Vocab data saved to {vocab_path}")

# ----------------- 총 학습 시간 -----------------
total_training_duration = time.time() - start_time
print(f"\n✅ Total Training Time: {total_training_duration:.2f} seconds ({total_training_duration/3600:.2f} hours)")
