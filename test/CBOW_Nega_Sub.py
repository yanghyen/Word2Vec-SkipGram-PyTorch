import nltk
from nltk.corpus import brown
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import Counter

nltk.download('brown')

# -----------------------------
# 1. Corpus 준비
# -----------------------------
sentences = brown.sents()
corpus = [w.lower() for sent in sentences for w in sent]

# 단어 빈도수
word_freq = Counter(corpus)
vocab = list(word_freq.keys())
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
total_count = len(corpus)

# -----------------------------
# 2. Subsampling
# -----------------------------
def subsample(corpus, threshold=1e-5):
    subsampled = []
    for word in corpus:
        freq = word_freq[word] / total_count
        prob_keep = (math.sqrt(freq / threshold) + 1) * (threshold / freq)
        if random.random() < prob_keep:
            subsampled.append(word)
    return subsampled

subsampled_corpus = subsample(corpus)

# -----------------------------
# 3. CBOW 데이터 생성
# -----------------------------
window_size = 2

def generate_cbow_data(corpus, window_size):
    data = []
    for i in range(window_size, len(corpus) - window_size):
        context = corpus[i-window_size:i] + corpus[i+1:i+window_size+1]
        target = corpus[i]
        data.append((context, target))
    return data

data = generate_cbow_data(subsampled_corpus, window_size)

# -----------------------------
# 4. Dataset + Negative Sampling
# -----------------------------
class CBOWDataset(torch.utils.data.Dataset):
    def __init__(self, data, word2idx, vocab_freq, neg_sample_size=5):
        self.data = data
        self.word2idx = word2idx
        self.neg_sample_size = neg_sample_size
        freq_sum = sum([f**0.75 for f in vocab_freq.values()])
        self.neg_dist = [vocab_freq[w]**0.75 / freq_sum for w in vocab_freq]
        self.vocab = list(vocab_freq.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idx = torch.tensor([self.word2idx[w] for w in context], dtype=torch.long)
        pos_target = torch.tensor(self.word2idx[target], dtype=torch.long)
        neg_samples = random.choices(self.vocab, weights=self.neg_dist, k=self.neg_sample_size)
        neg_samples_idx = torch.tensor([self.word2idx[w] for w in neg_samples], dtype=torch.long)
        return context_idx, pos_target, neg_samples_idx

dataset = CBOWDataset(data, word2idx, word_freq, neg_sample_size=5)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# 5. CBOW 모델 정의
# -----------------------------
class CBOWNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWNegSampling, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context_idxs, pos_target, neg_samples):
        context_vec = self.in_embed(context_idxs).mean(dim=1)
        pos_embeds = self.out_embed(pos_target)
        pos_score = torch.mul(context_vec, pos_embeds).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)
        neg_embeds = self.out_embed(neg_samples)
        neg_score = torch.bmm(neg_embeds, context_vec.unsqueeze(2)).squeeze()
        neg_loss = torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)
        return -(pos_loss + neg_loss).mean()

# -----------------------------
# 6. 학습
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 50
model = CBOWNegSampling(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)

for epoch in range(3):
    total_loss = 0
    for context_idxs, pos_target, neg_samples in dataloader:
        context_idxs, pos_target, neg_samples = context_idxs.to(device), pos_target.to(device), neg_samples.to(device)
        optimizer.zero_grad()
        loss = model(context_idxs, pos_target, neg_samples)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------------
# 7. 유사 단어 검색
# -----------------------------
def most_similar(word, model, word2idx, idx2word, top_k=5):
    vec = model.in_embed(torch.tensor([word2idx[word]]).to(device)).detach()
    embeddings = model.in_embed.weight
    similarities = F.cosine_similarity(vec, embeddings)
    top_idx = torch.topk(similarities, top_k+1).indices.tolist()
    top_idx.remove(word2idx[word])
    return [idx2word[i] for i in top_idx[:top_k]]

print("Most similar to 'bank':", most_similar("bank", model, word2idx, idx2word))

# -----------------------------
# 8. 아날로지 예제
# -----------------------------
def analogy(word_a, word_b, word_c, model, word2idx, idx2word, top_k=5):
    vec_a = model.in_embed(torch.tensor([word2idx[word_a]]).to(device)).detach()
    vec_b = model.in_embed(torch.tensor([word2idx[word_b]]).to(device)).detach()
    vec_c = model.in_embed(torch.tensor([word2idx[word_c]]).to(device)).detach()
    target_vec = vec_b - vec_a + vec_c
    embeddings = model.in_embed.weight
    similarities = F.cosine_similarity(target_vec, embeddings)
    top_idx = torch.topk(similarities, top_k+1).indices.tolist()
    top_idx = [i for i in top_idx if i != word2idx[word_c]]
    return [idx2word[i] for i in top_idx[:top_k]]

print("king - man + woman ≈", analogy("king", "man", "woman", model, word2idx, idx2word))

