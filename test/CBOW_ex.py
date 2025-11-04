import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import brown
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

sentences = brown.sents()
text = [w.lower() for sent in sentences for w in sent]
vocab = set(text)
vocab_size = len(vocab)

def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

data = []

for i in range(2, len(text) - 2):
    context = [text[i-2], text[i-1],
               text[i+1], text[i+2]]    # 타겟 워드 빼고 넣기 
    target = text[i]
    data.append((context, target))

class CBOWDataset(torch.utils.data.Dataset):
    def __init__(self, data, word_to_idx):
        self.data = data
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idx = torch.tensor([self.word_to_idx[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor(self.word_to_idx[target], dtype=torch.long)
        return context_idx, target_idx

class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128):
        super(CBOW, self).__init__()    # 해당 클래스를 torch.nn처럼 쓸 수 있음 
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.mean(dim=1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        return out

dataset = CBOWDataset(data, word_to_idx)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

embedding_dim = 10
model = CBOW(vocab_size, embedding_dim).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001)

loss_function = nn.CrossEntropyLoss()

for epoch in range(50):
    total_loss = 0
    for context_idxs, target in dataloader:
        context_idxs, target = context_idxs.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(context_idxs)
        loss = loss_function(output, target)
        
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step() 
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Let's see if our CBOW model works or not

print("*************************************************************************")

context = ['the', 'bank', 'of', 'the']
context_idxs = torch.tensor([[word_to_idx[w] for w in context]], dtype=torch.long).to(device)
logits = model(context_idxs)
predicted_idx = torch.argmax(logits, dim=1).item()
print("Predicted center word:", idx_to_word[predicted_idx])


