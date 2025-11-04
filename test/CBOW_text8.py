# cbow_text8.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# -----------------------
# 1. Text8 데이터 로딩
# -----------------------
import requests, zipfile, io

print("Downloading Text8 dataset...")
url = "http://mattmahoney.net/dc/text8.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
text = z.read("text8").decode('utf-8')
print("Text8 loaded!")

# -----------------------
# 2. 데이터 전처리
# -----------------------
tokens = text.lower().split()
vocab_count = Counter(tokens)
vocab = [word for word, freq in vocab_count.items() if freq >= 5]  # 최소 5회 등장
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

filtered_tokens = [w for w in tokens if w in word2idx]

# -----------------------
# 3. CBOW Dataset
# -----------------------
class CBOWDataset(Dataset):
    def __init__(self, tokens, word2idx, window_size=2, negative_samples=5):
        self.tokens = tokens
        self.word2idx = word2idx
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.data = self.create_pairs()
        self.vocab_indices = list(word2idx.values())
    
    def create_pairs(self):
        pairs = []
        for i in range(self.window_size, len(self.tokens) - self.window_size):
            context = [self.tokens[i - j] for j in range(self.window_size, 0, -1)] + \
                      [self.tokens[i + j] for j in range(1, self.window_size + 1)]
            target = self.tokens[i]
            pairs.append((context, target))
        return pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idx = torch.tensor([self.word2idx[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor(self.word2idx[target], dtype=torch.long)
        neg_samples = torch.tensor(self.get_negative_samples(target_idx), dtype=torch.long)
        return context_idx, target_idx, neg_samples

    def get_negative_samples(self, target_idx):
        neg_samples = []
        while len(neg_samples) < self.negative_samples:
            neg = random.choice(self.vocab_indices)
            if neg != target_idx:
                neg_samples.append(neg)
        return neg_samples

# -----------------------
# 4. CBOW 모델
# -----------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOW, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_linear = nn.Linear(embed_size, vocab_size)
    
    def forward(self, context_idxs):
        embeds = self.in_embed(context_idxs)  # batch x context x embed
        context_sum = embeds.sum(dim=1)       # batch x embed
        context_avg = context_sum / embeds.size(1)
        out = self.out_linear(context_avg)     # batch x vocab
        return out

# -----------------------
# 5. 학습 세팅
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

window_size = 5
negative_samples = 5
embedding_dim = 200
batch_size = 128
epochs = 50
learning_rate = 0.01

dataset = CBOWDataset(filtered_tokens, word2idx, window_size, negative_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CBOW(len(word2idx), embedding_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# -----------------------
# 6. 학습 루프
# -----------------------
for epoch in range(epochs):
    total_loss = 0
    for context_idx, target_idx, neg_samples in dataloader:
        context_idx, target_idx = context_idx.to(device), target_idx.to(device)

        optimizer.zero_grad()
        logits = model(context_idx)
        loss = loss_fn(logits, target_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# -----------------------
# 7. 유사 단어 검색
# -----------------------
def most_similar(word, top_k=5):
    if word not in word2idx:
        return []
    word_vec = model.in_embed(torch.tensor([word2idx[word]], device=device))
    sims = []
    for idx in range(len(word2idx)):
        if idx == word2idx[word]:
            continue
        other_vec = model.in_embed(torch.tensor([idx], device=device))
        sim = torch.cosine_similarity(word_vec, other_vec)
        sims.append((idx, sim.item()))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [(idx2word[idx], sim) for idx, sim in sims[:top_k]]

print("Most similar to 'king':", most_similar('king'))

# -----------------------
# 8. 아날로지 테스트
# -----------------------
def analogy(word_a, word_b, word_c, top_k=5):
    indices = [word2idx.get(w) for w in [word_a, word_b, word_c]]
    if None in indices:
        return []
    vecs = [model.in_embed(torch.tensor([i], device=device)) for i in indices]
    result_vec = vecs[1] - vecs[0] + vecs[2]
    sims = []
    for idx in range(len(word2idx)):
        if idx in indices:
            continue
        other_vec = model.in_embed(torch.tensor([idx], device=device))
        sim = torch.cosine_similarity(result_vec, other_vec)
        sims.append((idx, sim.item()))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [(idx2word[idx], sim) for idx, sim in sims[:top_k]]

print("king - man + woman ≈", analogy('king', 'man', 'woman'))

