import heapq

class Node:
    # 모든 노드가 고유 인덱스를 가질 수 있도록 수정
    def __init__(self, index=None, freq=0):
        self.index = index  # 단어 인덱스(0 ~ V-1) 또는 내부 노드 인덱스(V ~ 2V-2)
        self.freq = freq
        self.left = None
        self.right = None

class HuffmanTree:
    def __init__(self, word_freq):
        # 🌟 V 값 저장: 단어 노드의 개수 (내부 노드 인덱스 부여 시작점)
        self.vocab_size = len(word_freq) 
        
        # 🌟 내부 노드 인덱스 시작점 설정
        self.next_internal_index = self.vocab_size
        
        self.root = self._build_tree(word_freq)
        self.path_dict, self.code_dict = {}, {}
        
        # _generate_codes를 호출할 때, path에 index를 저장합니다.
        self._generate_codes(self.root, path=[], code=[])

    def _build_tree(self, word_freq):
        heap = []
        counter = 0 
        
        for idx, (word, freq) in enumerate(word_freq.items()):
            # Node 생성 시 word_idx 대신 index 사용 (0부터 V-1까지)
            heapq.heappush(heap, (freq, counter, Node(index=idx, freq=freq)))
            counter += 1
            
        while len(heap) > 1:
            freq1, _, n1 = heapq.heappop(heap)
            freq2, _, n2 = heapq.heappop(heap)
            
            # 🌟 내부 노드 생성 시 순차적 인덱스 부여
            merged_index = self.next_internal_index
            self.next_internal_index += 1
            
            merged = Node(index=merged_index, freq=freq1 + freq2)
            merged.left, merged.right = n1, n2
            
            heapq.heappush(heap, (merged.freq, counter, merged))
            counter += 1
            
        return heap[0][2] 

    def _generate_codes(self, node, path, code):
        # 노드가 단어(리프) 노드인지 확인 (내부 노드는 항상 index를 가짐)
        if node.left is None and node.right is None: 
            self.path_dict[node.index] = list(path)
            self.code_dict[node.index] = list(code)
            return
            
        new_path = list(path)
        if node.index >= self.vocab_size:
            new_path.append(node.index)
        
        if node.left:
            self._generate_codes(node.left, new_path, code + [0])
        if node.right:
            self._generate_codes(node.right, new_path, code + [1])
            
  
    # get_path, get_code 함수는 word_idx 대신 index를 받지만,
    # train.py에서 단어 인덱스로 호출하므로 그대로 둡니다.
    def get_path(self, word_idx):
        return self.path_dict[word_idx]

    def get_code(self, word_idx):
        return self.code_dict[word_idx]