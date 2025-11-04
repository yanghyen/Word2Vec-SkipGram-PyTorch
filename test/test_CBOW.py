import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from nltk.corpus import brown

sentences = brown.sents()

text = [w for sent in sentences for w in sent]

# vocablurary 구축
vocab = set(text)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

window_size = 2
data = []

for i in range(window_size, len(text) - window_size):
  context = [text[i - j - 1] for j in range(window_size)] + [text[i + j + 1] for j in range(window_size)]
  target = text[i]
  data.append((context, target))

class CBOWDataset(torch.utils.data.Dataset):
  def __init__(self, data, word2idx):
    self.data = data
    self.word2idx = word2idx

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    context, target = self.data[idx]
    context_idx = torch.tensor([self.word2idx[w] for w in context], dtype=torch.long)
    target_idx = torch.tensor(self.word2idx[target], dtype=torch.long)
    return context_idx, target_idx

dataset = CBOWDataset(data, word2idx)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

class CBOW(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear = nn.Linear(embedding_dim, vocab_size)

  def forward(self, context_idxs):
    # context의 embedding 평균
    embeds = self.embeddings(context_idxs)
    avg_embeds = embeds.mean(dim=1)
    out = self.linear(avg_embeds)
    return out

embedding_dim = 10
model = CBOW(len(vocab), embedding_dim)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
  total_loss = 0
  for context_idxs, target in dataloader:
    optimizer.zero_grad()
    output = model(context_idxs)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  if epoch % 10 == 0:
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

word = "money"
word_idx = torch.tensor([word2idx[word]])
embedding_vector = model.embeddings(word_idx).detach().numpy()
print(f"'{word}' 임베딩 벡터:", embedding_vector)

torch.save(model.state_dict(), "CBOWbyPyTorch.pth")
