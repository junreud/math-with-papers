"""
T5 Implementation based on "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
논문의 핵심 수식과 개념을 충실히 구현

주요 구성 요소:
1. Text-to-Text Transfer Transformer
2. Encoder-Decoder 아키텍처 (original Transformer 기반)
3. Span Corruption Pre-training
4. Unified Text-to-Text Format
5. Relative Position Encoding
6. 다양한 NLP 태스크의 통합 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional, Tuple, List, Dict
from enum import Enum


class T5TaskFormat:
    """
    T5의 핵심: 모든 NLP 태스크를 Text-to-Text 형식으로 통일
    """
    
    # Translation
    TRANSLATION = "translate English to German: {text}"
    
    # Summarization  
    SUMMARIZATION = "summarize: {text}"
    
    # Question Answering
    QA = "question: {question} context: {context}"
    
    # Text Classification
    CLASSIFICATION = "cola sentence: {sentence}"
    
    # Natural Language Inference
    NLI = "mnli premise: {premise} hypothesis: {hypothesis}"
    
    # Parsing
    PARSING = "parse: {sentence}"
    
    # Reading Comprehension
    READING_COMP = "trivia question: {question}"


class RelativePositionBias(nn.Module):
    """
    T5의 핵심 혁신: Relative Position Encoding
    절대 위치 대신 상대적 위치 관계를 학습
    
    논문 수식: 
    - Attention에 relative position bias 추가
    - A_ij = Q_i · K_j + b_{clip(i-j, -k, k)}
    """
    def __init__(
        self, 
        num_heads: int, 
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.bidirectional = bidirectional
        
        # Relative position bias embedding
        self.relative_attention_bias = nn.Embedding(
            relative_attention_num_buckets, num_heads
        )
    
    def _relative_position_bucket(
        self, 
        relative_position: torch.Tensor,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """
        T5 논문의 relative position bucketing 구현
        상대적 거리를 bucket으로 나누어 효율적으로 처리
        """
        relative_buckets = 0
        
        if self.bidirectional:
            num_buckets //= 2
            # 양수/음수 구분
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # Decoder의 경우 미래는 볼 수 없음
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # 가까운 거리는 더 세밀하게, 먼 거리는 더 coarse하게
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # 로그 스케일로 먼 거리 처리
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        
        return relative_buckets
    
    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """
        Relative position bias 계산
        
        Returns:
            bias: (1, num_heads, query_length, key_length)
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        
        # 상대적 위치 계산: i - j
        relative_position = memory_position - context_position
        
        # Bucket으로 변환
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        
        # Bias 계산
        bias = self.relative_attention_bias(relative_position_bucket)  # (query_len, key_len, num_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, query_len, key_len)
        
        return bias


class MultiHeadAttention(nn.Module):
    """
    T5 Multi-Head Attention with Relative Position Bias
    논문 수식: Attention(Q, K, V) = softmax((QK^T + bias) / sqrt(d_k))V
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
        has_relative_attention_bias: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V projections (T5는 bias 없음)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias (첫 번째 layer에만)
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = RelativePositionBias(num_heads)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model) 
            value: (batch_size, value_len, d_model)
            attention_mask: (batch_size, query_len, key_len)
            position_bias: (batch_size, num_heads, query_len, key_len)
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        
        # Linear projections
        Q = self.W_q(query)  # (batch_size, query_len, d_model)
        K = self.W_k(key)    # (batch_size, key_len, d_model)
        V = self.W_v(value)  # (batch_size, value_len, d_model)
        
        # Split into heads
        Q = Q.view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias
        if self.has_relative_attention_bias:
            position_bias = self.relative_attention_bias(query_len, key_len)
        
        if position_bias is not None:
            scores += position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(1) == 0, 
                float('-inf')
            )
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Concat heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        # Final projection
        output = self.W_o(output)
        
        return output, position_bias


class FeedForward(nn.Module):
    """
    T5 Position-wise Feed-Forward Network
    논문 수식: FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
    T5는 ReLU 사용 (GELU 아님)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff, bias=False)
        self.W_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        T5 FFN: ReLU(xW_1)W_2 (bias 없음)
        """
        hidden = F.relu(self.W_1(x))
        hidden = self.dropout(hidden)
        output = self.W_2(hidden)
        return output


class T5LayerNorm(nn.Module):
    """
    T5-style Layer Normalization
    논문: RMSNorm과 유사하지만 center는 하지 않음
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # T5는 mean을 빼지 않고 variance로만 normalize
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Scale
        return self.weight * hidden_states


class T5Block(nn.Module):
    """
    T5 Transformer Block
    T5는 pre-norm structure 사용
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        has_relative_attention_bias: bool = False,
        is_decoder: bool = False
    ):
        super().__init__()
        self.is_decoder = is_decoder
        
        # Self-attention
        self.layer = nn.ModuleList()
        self.layer.append(
            nn.ModuleList([
                T5LayerNorm(d_model),
                MultiHeadAttention(
                    d_model, num_heads, dropout, has_relative_attention_bias
                ),
                nn.Dropout(dropout)
            ])
        )
        
        # Cross-attention (decoder only)
        if is_decoder:
            self.layer.append(
                nn.ModuleList([
                    T5LayerNorm(d_model),
                    MultiHeadAttention(d_model, num_heads, dropout, False),
                    nn.Dropout(dropout)
                ])
            )
        
        # Feed-forward
        self.layer.append(
            nn.ModuleList([
                T5LayerNorm(d_model),
                FeedForward(d_model, d_ff, dropout),
                nn.Dropout(dropout)
            ])
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # Self-attention
        layer_module = self.layer[0]
        norm_hidden_states = layer_module[0](hidden_states)
        attention_output, position_bias = layer_module[1](
            norm_hidden_states, norm_hidden_states, norm_hidden_states,
            attention_mask, position_bias
        )
        hidden_states = hidden_states + layer_module[2](attention_output)
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            layer_module = self.layer[1]
            norm_hidden_states = layer_module[0](hidden_states)
            attention_output, encoder_decoder_position_bias = layer_module[1](
                norm_hidden_states, encoder_hidden_states, encoder_hidden_states,
                encoder_attention_mask, encoder_decoder_position_bias
            )
            hidden_states = hidden_states + layer_module[2](attention_output)
        
        # Feed-forward
        ff_layer_index = 2 if self.is_decoder else 1
        layer_module = self.layer[ff_layer_index]
        norm_hidden_states = layer_module[0](hidden_states)
        ff_output = layer_module[1](norm_hidden_states)
        hidden_states = hidden_states + layer_module[2](ff_output)
        
        return hidden_states, position_bias, encoder_decoder_position_bias


class T5Stack(nn.Module):
    """
    T5 Encoder 또는 Decoder Stack
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        is_decoder: bool = False
    ):
        super().__init__()
        self.is_decoder = is_decoder
        
        # Transformer blocks
        self.block = nn.ModuleList([
            T5Block(
                d_model, num_heads, d_ff, dropout,
                has_relative_attention_bias=(i == 0),  # 첫 번째 layer만
                is_decoder=is_decoder
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        position_bias = None
        encoder_decoder_position_bias = None
        
        for layer_module in self.block:
            hidden_states, position_bias, encoder_decoder_position_bias = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias
            )
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class SpanCorruption:
    """
    T5의 Pre-training: Span Corruption
    논문에서 사용된 BERT-style masked language modeling의 변형
    """
    
    @staticmethod
    def corrupt_spans(
        text: List[int],
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0
    ) -> Tuple[List[int], List[int]]:
        """
        T5 Span Corruption 구현
        
        Args:
            text: 원본 토큰 리스트
            noise_density: corruption 비율 (기본 15%)
            mean_noise_span_length: 평균 span 길이 (기본 3)
            
        Returns:
            corrupted_text: corruption된 입력 텍스트
            target_text: 복원해야 할 타겟 텍스트
        """
        if len(text) == 0:
            return [], []
        
        # 마스킹할 토큰 수 계산
        num_noise_tokens = int(round(len(text) * noise_density))
        
        # 평균 span 길이를 기반으로 span 수 계산
        num_noise_spans = max(1, round(num_noise_tokens / mean_noise_span_length))
        
        # 각 span 길이 결정
        noise_span_lengths = []
        remaining_tokens = num_noise_tokens
        
        for _ in range(num_noise_spans - 1):
            # 지수분포에서 샘플링
            span_length = max(1, int(random.expovariate(1.0 / mean_noise_span_length)))
            span_length = min(span_length, remaining_tokens - (num_noise_spans - len(noise_span_lengths) - 1))
            noise_span_lengths.append(span_length)
            remaining_tokens -= span_length
        
        # 마지막 span
        noise_span_lengths.append(max(1, remaining_tokens))
        
        # 랜덤 위치에 span 배치
        num_nonnoise_tokens = len(text) - num_noise_tokens
        noise_token_indices = set()
        
        for span_length in noise_span_lengths:
            # 가능한 시작 위치들
            possible_starts = []
            for i in range(len(text) - span_length + 1):
                if not any(j in noise_token_indices for j in range(i, i + span_length)):
                    possible_starts.append(i)
            
            if possible_starts:
                start_index = random.choice(possible_starts)
                for i in range(start_index, start_index + span_length):
                    noise_token_indices.add(i)
        
        # 입력과 타겟 생성
        corrupted_tokens = []
        target_tokens = []
        
        sentinel_id = 0
        i = 0
        
        while i < len(text):
            if i in noise_token_indices:
                # 연속된 noise span 찾기
                span_start = i
                while i < len(text) and i in noise_token_indices:
                    i += 1
                span_end = i
                
                # Sentinel 토큰 추가
                sentinel_token = f"<extra_id_{sentinel_id}>"
                corrupted_tokens.append(sentinel_token)
                
                # 타겟에 sentinel + 원본 토큰들 추가
                target_tokens.append(sentinel_token)
                target_tokens.extend(text[span_start:span_end])
                
                sentinel_id += 1
            else:
                corrupted_tokens.append(text[i])
                i += 1
        
        # 타겟 끝에 종료 토큰
        target_tokens.append("<extra_id_{}>".format(sentinel_id))
        
        return corrupted_tokens, target_tokens


class T5Model(nn.Module):
    """
    T5: Text-to-Text Transfer Transformer
    
    논문의 핵심 아이디어:
    1. 모든 NLP 태스크를 text-to-text 형식으로 통일
    2. Encoder-Decoder 아키텍처
    3. Span corruption pre-training
    4. Relative position encoding
    """
    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Shared embedding
        self.shared = nn.Embedding(vocab_size, d_model)
        
        # Encoder
        self.encoder = T5Stack(
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            is_decoder=False
        )
        
        # Decoder
        self.decoder = T5Stack(
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            is_decoder=True
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """T5 weight initialization"""
        # Embedding 초기화
        self.shared.weight.data.normal_(mean=0.0, std=1.0)
        
        # Linear layer 초기화
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def get_input_embeddings(self):
        return self.shared
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def create_decoder_attention_mask(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """
        Decoder용 causal mask 생성
        """
        batch_size, seq_len = decoder_input_ids.size()
        
        # Causal mask (하삼각행렬)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=decoder_input_ids.device))
        
        # Padding mask
        padding_mask = (decoder_input_ids != 0).float()
        
        # 결합
        attention_mask = causal_mask.unsqueeze(0) * padding_mask.unsqueeze(1)
        
        return attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, src_len) - encoder input
            attention_mask: (batch_size, src_len) - encoder padding mask
            decoder_input_ids: (batch_size, tgt_len) - decoder input
            decoder_attention_mask: (batch_size, tgt_len) - decoder mask
            labels: (batch_size, tgt_len) - target for loss computation
        """
        
        # Encoder
        encoder_embeddings = self.shared(input_ids)
        encoder_outputs = self.encoder(
            encoder_embeddings,
            attention_mask=attention_mask
        )
        
        # Decoder
        if decoder_input_ids is not None:
            decoder_embeddings = self.shared(decoder_input_ids)
            
            if decoder_attention_mask is None:
                decoder_attention_mask = self.create_decoder_attention_mask(decoder_input_ids)
            
            decoder_outputs = self.decoder(
                decoder_embeddings,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=attention_mask
            )
            
            # Language modeling head
            logits = self.lm_head(decoder_outputs)
        else:
            logits = None
        
        # Loss computation
        loss = None
        if labels is not None and logits is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> torch.Tensor:
        """
        Text generation (간단한 greedy decoding)
        """
        self.eval()
        
        with torch.no_grad():
            # Encoder
            encoder_embeddings = self.shared(input_ids)
            encoder_outputs = self.encoder(
                encoder_embeddings,
                attention_mask=attention_mask
            )
            
            # Decoder 초기화
            batch_size = input_ids.size(0)
            decoder_input_ids = torch.zeros(
                batch_size, 1, dtype=torch.long, device=input_ids.device
            )
            
            for _ in range(max_length):
                decoder_embeddings = self.shared(decoder_input_ids)
                decoder_attention_mask = self.create_decoder_attention_mask(decoder_input_ids)
                
                decoder_outputs = self.decoder(
                    decoder_embeddings,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=attention_mask
                )
                
                logits = self.lm_head(decoder_outputs)
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                
                # EOS 토큰이면 종료
                if next_token.item() == 1:  # EOS token id
                    break
        
        return decoder_input_ids


def create_t5_model(model_size: str = "base") -> T5Model:
    """
    T5 모델 생성 (다양한 크기)
    """
    configs = {
        "small": {
            "d_model": 512,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "num_heads": 8,
            "d_ff": 2048
        },
        "base": {
            "d_model": 768,
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
            "num_heads": 12,
            "d_ff": 3072
        },
        "large": {
            "d_model": 1024,
            "num_encoder_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 16,
            "d_ff": 4096
        },
        "3b": {
            "d_model": 1024,
            "num_encoder_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 32,
            "d_ff": 16384
        },
        "11b": {
            "d_model": 1024,
            "num_encoder_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 128,
            "d_ff": 65536
        }
    }
    
    config = configs.get(model_size, configs["base"])
    return T5Model(**config)


# 사용 예시
if __name__ == "__main__":
    """
    T5 논문 구현 요약:
    
    1. 핵심 아이디어:
       - Text-to-Text Transfer Transformer
       - 모든 NLP 태스크를 "텍스트 입력 → 텍스트 출력" 형식으로 통일
       - Transfer learning의 극한 탐구
    
    2. 아키텍처:
       - Encoder-Decoder Transformer
       - Relative Position Encoding
       - Pre-normalization (T5LayerNorm)
       - ReLU activation in FFN
    
    3. Pre-training:
       - Span Corruption objective
       - 15% corruption rate, 평균 3 토큰 span
       - C4 (Colossal Clean Crawled Corpus) 데이터
    
    4. Fine-tuning:
       - Task-specific prefix 추가
       - 모든 태스크를 동일한 형식으로 처리
       - Multi-task learning 가능
    """
    
    print("=" * 60)
    print("T5: Text-to-Text Transfer Transformer")
    print("=" * 60)
    
    # 모델 생성
    model = create_t5_model("base")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model configuration:")
    print(f"- d_model: {model.d_model}")
    print(f"- vocab_size: {model.vocab_size}")
    
    # 예시 입력 (Text-to-Text format)
    batch_size = 2
    src_len = 32
    tgt_len = 16
    vocab_size = 32128
    
    # "translate English to German: Hello world" 같은 형식
    input_ids = torch.randint(1, vocab_size, (batch_size, src_len))
    decoder_input_ids = torch.randint(1, vocab_size, (batch_size, tgt_len))
    labels = torch.randint(1, vocab_size, (batch_size, tgt_len))
    
    print(f"\nExample forward pass:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")
    
    with torch.no_grad():
        logits, loss = model(input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        print(f"Output logits shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")
    
    # Span corruption 예시
    print(f"\nSpan Corruption example:")
    span_corruptor = SpanCorruption()
    text = list(range(10, 30))  # [10, 11, 12, ..., 29]
    corrupted, target = span_corruptor.corrupt_spans(text)
    print(f"Original: {text}")
    print(f"Corrupted: {corrupted}")
    print(f"Target: {target}")
    
    print("\n" + "=" * 60)
    print("T5 구현 완료!")
    print("주요 특징:")
    print("1. ✅ Text-to-Text unified format")
    print("2. ✅ Encoder-Decoder architecture")
    print("3. ✅ Relative Position Encoding")
    print("4. ✅ Span Corruption pre-training")
    print("5. ✅ T5LayerNorm (no centering)")
    print("6. ✅ Multi-task learning support")
    print("=" * 60)