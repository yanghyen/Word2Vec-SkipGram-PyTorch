import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import brown
import numpy as np

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Brown 코퍼스 불러오기
sentences = brown.sents()
text = [w.lower() for sent in sentences for w in sent]  # 소문자로 통일

# Vocabulary 구축
vocab = set(text)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# Skip-gram 데이터셋
window_size = 5
data_sg = []
for i in range(window_size, len(text) - window_size):
    center = text[i]
    context = [text[i - j - 1] for j in range(window_size)] + [text[i + j + 1] for j in range(window_size)]
    for w in context:
        data_sg.append((center, w))

class SkipGramDataset(Dataset):
    def __init__(self, data, word2idx):
        self.data = data
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        center_idx = torch.tensor(self.word2idx[center], dtype=torch.long)
        context_idx = torch.tensor(self.word2idx[context], dtype=torch.long)
        return center_idx, context_idx

dataset_sg = SkipGramDataset(data_sg, word2idx)
dataloader_sg = DataLoader(dataset_sg, batch_size=512, shuffle=True)

print("Batch count:", len(dataloader_sg))

# Skip-gram 모델
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_idx):
        embed = self.embeddings(center_idx)
        out = self.linear(embed)
        return out

embedding_dim = 50
model_sg = SkipGram(len(vocab), embedding_dim).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_sg.parameters(), lr=0.01)

# 학습
for epoch in range(50):
    total_loss = 0
    for i, (center_idx, context_idx) in enumerate(dataloader_sg):
        center_idx = center_idx.to(device)
        context_idx = context_idx.to(device)

        optimizer.zero_grad()
        output = model_sg(center_idx)
        loss = loss_fn(output, context_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 1000 == 0:
            print(f"Epoch {epoch} Step {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} finished, Total Loss: {total_loss:.4f}")

# 임베딩 벡터 확인
def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)

money_vec = model_sg.embeddings(torch.tensor([word2idx["money"]]).to(device)).detach().cpu().numpy()[0]
finance_vec = model_sg.embeddings(torch.tensor([word2idx["finance"]]).to(device)).detach().cpu().numpy()[0]

sim = cosine_similarity(money_vec, finance_vec)
print("money ~ finance 유사도:", sim)

# 모델 저장
torch.save({
    "model_state_dict": model_sg.state_dict(),
    "word2idx": word2idx,
    "idx2word": idx2word
}, "SkipGrambyPyTorch.pth")

