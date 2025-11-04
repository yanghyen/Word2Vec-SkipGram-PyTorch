from datasets import load_dataset
import os

# 1. Hugging Face 데이터셋 캐시의 최상위 경로를 지정합니다.
CACHE_DIR = "/home/ssai/Workspace/Word2Vec_repo/data/pretrain/huggingface_cache"

# 2. 최종 코퍼스 파일이 저장될 경로를 지정합니다.
OUTPUT_PATH = "data/pretrain/word2vec_corpus_hf_half.txt"

# 3. 데이터셋 로드 (캐시된 파일을 자동으로 사용합니다)
print("📘 캐시된 데이터셋 로드 중...")
try:
    # ds는 DatasetDict 객체입니다. (예: {'train': Dataset(...)} )
    ds = load_dataset("lsb/enwiki20230101", cache_dir=CACHE_DIR)
except Exception as e:
    print(f"❌ 데이터셋 로드 오류. CACHE_DIR을 확인해주세요: {e}")
    ds = load_dataset("lsb/enwiki20230101")

ds_train = ds['train']
total_docs = len(ds_train)

half_docs = total_docs // 3  

ds_to_process = ds_train[:half_docs]['text'] 

print(f"총 {total_docs:,}개의 문서 중 {half_docs:,}개만 로드 및 처리 예정.")
print(f"문서를 {OUTPUT_PATH} 파일로 순차적으로 내보내는 중...")

write_count = 0
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for i, text in enumerate(ds_to_process): 
        stripped_text = text.strip()
        
        if stripped_text:
            f.write(stripped_text)
            f.write('\n\n') 
            write_count += 1
        
        if (i + 1) % 100000 == 0:
            print(f"    - {i+1:,}번째 문서까지 처리 완료...")

print("\n---")
print(f"코퍼스 파일 생성 완료: {OUTPUT_PATH}")
print(f"Total documents exported: {write_count:,}")
os.system(f"ls -lh {OUTPUT_PATH}") 