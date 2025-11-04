"""
UL2 Implementation based on "UL2: Unifying Language Learning Paradigms"
논문의 핵심 수식과 개념을 충실히 구현

주요 구성 요소:
1. Mixture of Denoisers (MoD) - 다양한 denoising 작업의 조합
2. R-Denoiser (Regular span corruption)
3. S-Denoiser (Sequential denoising) 
4. X-Denoiser (Extreme denoising)
5. Decoder-only Transformer 아키텍처
6. Mode switching을 위한 특별 토큰들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional, Tuple, List, Dict
from enum import Enum


class DenoisingMode(Enum):
    """
    UL2의 세 가지 denoising 모드
    논문에서 제안된 서로 다른 학습 패러다임
    """
    R_DENOISER = "R"  # Regular span corruption (BERT-style)
    S_DENOISER = "S"  # Sequential denoising (GPT-style)
    X_DENOISER = "X"  # Extreme denoising (long sequences)


class SpecialTokens:
    """
    UL2에서 사용하는 특별 토큰들
    Mode switching과 span corruption을 위한 토큰들
    """
    # Mode tokens
    R_MODE = "<R>"      # R-Denoiser mode
    S_MODE = "<S>"      # S-Denoiser mode  
    X_MODE = "<X>"      # X-Denoiser mode
    
    # Sentinel tokens for span corruption
    SENTINEL_0 = "<extra_id_0>"
    SENTINEL_1 = "<extra_id_1>"
    SENTINEL_2 = "<extra_id_2>"
    # ... 최대 100개까지
    
    # Special tokens
    PAD = "<pad>"
    EOS = "</s>"
    UNK = "<unk>"


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention
    논문 수식: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    UL2는 PaLM 스타일의 decoder-only 구조 사용
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        논문 수식: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
        Args:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            attention_mask: (batch_size, 1, seq_len, seq_len)
            is_causal: decoder-only 모델이므로 기본적으로 causal
        """
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask (decoder-only)
        if is_causal:
            seq_len = Q.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Additional attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # softmax(QK^T / sqrt(d_k))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # softmax(...)V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, 1, 1, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output, _ = self.scaled_dot_product_attention(
            Q, K, V, attention_mask, is_causal
        )
        
        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks with SwiGLU activation
    논문에서 PaLM 스타일의 SwiGLU 활성화 함수 사용
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU는 두 개의 linear projection 필요
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU activation: SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)
        where Swish(x) = x * sigmoid(x)
        """
        gate = F.silu(self.W_gate(x))  # SiLU = Swish
        up = self.W_up(x)
        hidden = gate * up  # Element-wise multiplication
        output = self.W_down(self.dropout(hidden))
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    논문에서 PaLM 스타일의 RMSNorm 사용
    수식: RMSNorm(x) = x / RMS(x) * g
    where RMS(x) = sqrt(mean(x^2) + ε)
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        # RMS(x) = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # x / RMS(x) * weight
        normalized = x / rms * self.weight
        
        return normalized


class TransformerBlock(nn.Module):
    """
    UL2 Transformer Block (Decoder-only)
    PaLM 스타일의 구조: Pre-normalization + SwiGLU + RMSNorm
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pre-normalization structure:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
        """
        # Pre-norm attention
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, attention_mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x


class SpanCorruptor:
    """
    UL2의 핵심: Mixture of Denoisers를 위한 Span Corruption
    R, S, X 각각 다른 corruption 전략 사용
    """
    
    @staticmethod
    def r_denoiser_corruption(
        text: List[int], 
        corruption_rate: float = 0.15,
        mean_span_length: float = 3.0
    ) -> Tuple[List[int], List[int]]:
        """
        R-Denoiser: BERT-style span corruption
        - 15% corruption rate
        - 평균 3 토큰 span length
        - 연속된 span들을 마스킹
        """
        if len(text) == 0:
            return [], []
            
        # 마스킹할 총 토큰 수 계산
        num_to_mask = max(1, int(len(text) * corruption_rate))
        
        # Span 생성
        spans = []
        remaining_tokens = num_to_mask
        
        while remaining_tokens > 0:
            # Poisson distribution에서 span 길이 샘플링
            span_length = max(1, int(random.expovariate(1.0 / mean_span_length)))
            span_length = min(span_length, remaining_tokens)
            
            # 랜덤 시작 위치
            start_pos = random.randint(0, len(text) - span_length)
            spans.append((start_pos, start_pos + span_length))
            remaining_tokens -= span_length
        
        # Span 정렬 및 병합
        spans.sort()
        merged_spans = []
        for start, end in spans:
            if merged_spans and start <= merged_spans[-1][1]:
                # 겹치는 span 병합
                merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
            else:
                merged_spans.append((start, end))
        
        # 입력과 타겟 생성
        input_tokens = []
        target_tokens = []
        sentinel_id = 0
        
        last_end = 0
        for start, end in merged_spans:
            # 마스킹되지 않은 부분 추가
            input_tokens.extend(text[last_end:start])
            
            # Sentinel 토큰 추가
            sentinel_token = f"<extra_id_{sentinel_id}>"
            input_tokens.append(sentinel_token)
            
            # 타겟에 sentinel + 원본 토큰들 추가
            target_tokens.append(sentinel_token)
            target_tokens.extend(text[start:end])
            
            sentinel_id += 1
            last_end = end
        
        # 마지막 부분 추가
        input_tokens.extend(text[last_end:])
        target_tokens.append("</s>")
        
        return input_tokens, target_tokens
    
    @staticmethod
    def s_denoiser_corruption(
        text: List[int],
        prefix_length: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        S-Denoiser: Sequential denoising (GPT-style)
        - 시퀀스의 prefix를 주고 나머지를 생성
        - Auto-regressive language modeling
        """
        if len(text) == 0:
            return [], []
            
        if prefix_length is None:
            # 랜덤하게 prefix 길이 선택 (50-90%)
            prefix_length = random.randint(
                int(len(text) * 0.5), 
                int(len(text) * 0.9)
            )
        
        prefix_length = min(prefix_length, len(text) - 1)
        
        input_tokens = text[:prefix_length]
        target_tokens = text[prefix_length:] + ["</s>"]
        
        return input_tokens, target_tokens
    
    @staticmethod
    def x_denoiser_corruption(
        text: List[int],
        corruption_rate: float = 0.5,
        mean_span_length: float = 32.0
    ) -> Tuple[List[int], List[int]]:
        """
        X-Denoiser: Extreme denoising
        - 높은 corruption rate (50%)
        - 긴 span length (평균 32)
        - 더 challenging한 denoising 작업
        """
        return SpanCorruptor.r_denoiser_corruption(
            text, corruption_rate, mean_span_length
        )


class UL2Model(nn.Module):
    """
    UL2: Unifying Language Learning Paradigms
    
    논문의 핵심 아이디어:
    1. Mixture of Denoisers (MoD) - 다양한 denoising 작업을 하나의 모델로
    2. Mode switching - 특별 토큰으로 학습 모드 지정
    3. Decoder-only 아키텍처
    4. 다양한 NLP 태스크에 적용 가능
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Embedding (learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_attention_mask(
        self, 
        input_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Padding mask 생성
        """
        mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(1)
        return mask.float()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) - for computing loss
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


class UL2Trainer:
    """
    UL2 모델 훈련을 위한 트레이너
    Mixture of Denoisers 구현
    """
    def __init__(
        self,
        model: UL2Model,
        tokenizer,  # 실제 구현시 tokenizer 필요
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.span_corruptor = SpanCorruptor()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Denoiser 비율 (논문에서 제안된 비율)
        self.denoiser_ratios = {
            DenoisingMode.R_DENOISER: 0.25,  # 25%
            DenoisingMode.S_DENOISER: 0.25,  # 25% 
            DenoisingMode.X_DENOISER: 0.50   # 50%
        }
    
    def sample_denoising_mode(self) -> DenoisingMode:
        """논문의 비율에 따라 denoising 모드 샘플링"""
        rand = random.random()
        
        if rand < self.denoiser_ratios[DenoisingMode.R_DENOISER]:
            return DenoisingMode.R_DENOISER
        elif rand < (self.denoiser_ratios[DenoisingMode.R_DENOISER] + 
                    self.denoiser_ratios[DenoisingMode.S_DENOISER]):
            return DenoisingMode.S_DENOISER
        else:
            return DenoisingMode.X_DENOISER
    
    def prepare_training_data(
        self, 
        texts: List[str]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, DenoisingMode]]:
        """
        훈련 데이터 준비 - 다양한 denoising 작업 생성
        """
        training_examples = []
        
        for text in texts:
            # 토크나이징 (실제 구현시 필요)
            tokens = self.tokenizer.encode(text)  # 가상의 토크나이저
            
            # 랜덤하게 denoising 모드 선택
            mode = self.sample_denoising_mode()
            
            # 해당 모드에 따라 corruption 적용
            if mode == DenoisingMode.R_DENOISER:
                input_tokens, target_tokens = self.span_corruptor.r_denoiser_corruption(tokens)
                mode_token = SpecialTokens.R_MODE
            elif mode == DenoisingMode.S_DENOISER:
                input_tokens, target_tokens = self.span_corruptor.s_denoiser_corruption(tokens)
                mode_token = SpecialTokens.S_MODE
            else:  # X_DENOISER
                input_tokens, target_tokens = self.span_corruptor.x_denoiser_corruption(tokens)
                mode_token = SpecialTokens.X_MODE
            
            # Mode token을 맨 앞에 추가
            input_tokens = [mode_token] + input_tokens
            
            # 텐서로 변환
            input_ids = torch.tensor(input_tokens, dtype=torch.long)
            labels = torch.tensor(target_tokens, dtype=torch.long)
            
            training_examples.append((input_ids, labels, mode))
        
        return training_examples
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """단일 훈련 스텝"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, loss = self.model(input_ids, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()


def create_ul2_model(
    model_size: str = "base"
) -> UL2Model:
    """
    UL2 모델 생성 (다양한 크기)
    """
    configs = {
        "small": {
            "vocab_size": 32000,
            "d_model": 512,
            "num_layers": 8,
            "num_heads": 8,
            "d_ff": 2048,
            "max_seq_len": 1024
        },
        "base": {
            "vocab_size": 32000,
            "d_model": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "d_ff": 4096,
            "max_seq_len": 2048
        },
        "large": {
            "vocab_size": 32000,
            "d_model": 1536,
            "num_layers": 32,
            "num_heads": 24,
            "d_ff": 6144,
            "max_seq_len": 2048
        }
    }
    
    config = configs.get(model_size, configs["base"])
    return UL2Model(**config)


# 사용 예시
if __name__ == "__main__":
    """
    UL2 논문 구현 요약:
    
    1. 핵심 아이디어:
       - Mixture of Denoisers: R, S, X 세 가지 denoising 작업
       - Mode switching: 특별 토큰으로 학습 모드 지정
       - 다양한 학습 패러다임을 하나의 모델로 통합
    
    2. 아키텍처:
       - Decoder-only Transformer (PaLM 스타일)
       - RMSNorm, SwiGLU activation
       - Pre-normalization structure
    
    3. Denoising 작업:
       - R-Denoiser: BERT-style span corruption (15% corruption)
       - S-Denoiser: GPT-style sequential denoising
       - X-Denoiser: Extreme denoising (50% corruption)
    
    4. 훈련:
       - Mixed ratio: R(25%), S(25%), X(50%)
       - Mode token을 통한 작업 구분
       - 다양한 NLP 태스크에 일반화
    """
    
    print("=" * 60)
    print("UL2: Unifying Language Learning Paradigms")
    print("=" * 60)
    
    # 모델 생성
    model = create_ul2_model("base")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model configuration:")
    print(f"- d_model: {model.d_model}")
    print(f"- num_layers: {len(model.transformer_blocks)}")
    print(f"- max_seq_len: {model.max_seq_len}")
    
    # 예시 입력
    batch_size = 2
    seq_len = 128
    vocab_size = 32000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nExample forward pass:")
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits, loss = model(input_ids, labels)
        print(f"Output logits shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("UL2 구현 완료!")
    print("주요 특징:")
    print("1. ✅ Mixture of Denoisers (R, S, X)")
    print("2. ✅ Mode switching with special tokens")
    print("3. ✅ Decoder-only Transformer architecture") 
    print("4. ✅ RMSNorm and SwiGLU activation")
    print("5. ✅ Span corruption for different denoising tasks")
    print("=" * 60)