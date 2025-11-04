import pickle

vocab = pickle.load(open("./runs/vocab.pkl", "rb"))

# 단어 → 인덱스
print("word2idx sample:", list(vocab["word2idx"].keys())[:50])

# 인덱스 → 단어 (dict이면 keys로 정렬)
idx2word_dict = vocab["idx2word"]
sorted_keys = sorted(idx2word_dict.keys())  # 0,1,2,... 순서
print("idx2word sample:", [idx2word_dict[k] for k in sorted_keys[:50]])
