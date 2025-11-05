import pickle
import os

VOCAB_PATH = os.path.join("runs", "vocab_ns.pkl")

try:
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
    
    word2idx = vocab_data["word2idx"]
    vocab_size = len(word2idx)
    
    print(f"로드된 Vocab 크기: {vocab_size:,}")
    
    # <unk> 토큰 확인
    if '<unk>' in word2idx:
        print(f"✅ '<unk>' 토큰이 존재합니다. 인덱스: {word2idx['<unk>']}")
    else:
        print("❌ '<unk>' 토큰이 존재하지 않습니다. 데이터셋을 다시 생성해야 합니다.")

    # Vocab의 처음 몇 개 항목 출력 (예시)
    print("\nVocab 일부 (단어: 인덱스):")
    for i, (word, idx) in enumerate(word2idx.items()):
        if i < 5:
            print(f"  {word}: {idx}")
        elif word == '<unk>':
             print(f"  <unk>: {idx}")

except FileNotFoundError:
    print(f"오류: Vocab 파일 경로를 찾을 수 없습니다: {VOCAB_PATH}")
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")