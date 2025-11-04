import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)            # 중심단어 벡터 저장
        # self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)         # 허프만 트리 돌 때를 위해 늘림
        self.out_embeddings = nn.Embedding(vocab_size * 2 - 1, embedding_dim)   # 주변단어 벡터 저장

        init_range = 0.5 / embedding_dim                                        # 초기화 전략: 임베딩 행렬의 값을 [-init_range, init_range] 범위 내의 랜덤한 값으로 초기화
        nn.init.uniform_(self.in_embeddings.weight, -init_range, init_range)    
        # nn.init.constant_(self.out_embeddings.weight, 0)
        nn.init.uniform_(self.out_embeddings.weight, -init_range, init_range)   

    def forward_ns(self, center_idx, context_idx, neg_idx):
        center = self.in_embeddings(center_idx)          
        context_pos = self.out_embeddings(context_idx)  
        context_neg = self.out_embeddings(neg_idx)      
        
        pos_score = torch.sum(center * context_pos, dim=1)
        pos_loss = F.logsigmoid(pos_score)
        
        neg_score = (context_neg * center.unsqueeze(1)).sum(dim=2)
        
        neg_loss = F.logsigmoid(-neg_score).sum(1) 

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def forward_hs(self, center_idx, path_idx, code_tensor, path_masks):
        """
        path_idx: List[List[int]] - 각 샘플의 허프만 경로
        code_tensor: List[List[float]] - 각 샘플의 코드 (0 or 1)
        """
        center = self.in_embeddings(center_idx)
        device = center.device
        
        batch_size = len(center_idx)
        
        # 벡터화 연산
        context_nodes = self.out_embeddings(path_idx)  # [B, max_len, D]
        center_expanded = center.unsqueeze(1)  # [B, 1, D]
        
        score = (context_nodes * center_expanded).sum(-1)  # [B, max_len]
        sign = 2 * code_tensor - 1
        
        log_prob = F.logsigmoid(score * sign)
        log_prob = log_prob * path_masks  # 패딩 마스킹
        
        loss = -log_prob.sum() / path_masks.sum()
        return loss
