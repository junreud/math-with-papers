"""
BERT Implementation based on "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
논문의 핵심 수식과 개념을 충실히 구현

주요 구성 요소:
1. Bidirectional Transformer Encoder (GPT와 달리 양방향)
2. Masked Language Model (MLM) Pre-training
3. Next Sentence Prediction (NSP) Pre-training
4. Token/Segment/Position Embeddings
5. Fine-tuning for downstream tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import random


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention
    논문 수식: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    BERT는 bidirectional이므로 causal mask 없음
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        논문 수식: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        BERT는 양방향이므로 모든 토큰이 서로를 볼 수 있음 (causal mask 없음)
        """
        # Q, K, V: (batch_size, num_heads, seq_len, d_k)
        
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Attention mask (padding mask만 적용, causal mask 없음)
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
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, 1, 1, seq_len) - padding mask
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
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, attention_mask)
        
        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final projection
        output = self.W_o(attn_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    논문 수식: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    BERT는 GELU activation 사용 (ReLU 대신)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        BERT는 GELU activation 사용
        GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of standard Gaussian
        """
        return self.W_2(self.dropout(F.gelu(self.W_1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer (BERT의 기본 building block)
    논문 구조:
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Position-wise FFN + Residual + LayerNorm
    
    GPT와 달리 양방향 attention 사용
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
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: padding mask
        """
        # Multi-Head Attention + Residual + LayerNorm
        attn_output = self.attention(x, attention_mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # Feed-Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_output))
        
        return x


class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings
    논문 수식: E = Token Embeddings + Segment Embeddings + Position Embeddings
    
    BERT의 3가지 embedding:
    1. Token Embeddings: WordPiece embeddings (30,000 vocab)
    2. Segment Embeddings: 문장 A와 B를 구분 (NSP task용)
    3. Position Embeddings: Learned positional embeddings
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Token Embeddings (WordPiece)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Segment Embeddings (0 for sentence A, 1 for sentence B)
        self.segment_embedding = nn.Embedding(2, d_model)
        
        # 3. Position Embeddings (learned, not sinusoidal like original Transformer)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices
            segment_ids: (batch_size, seq_len) - segment indices (0 or 1)
            position_ids: (batch_size, seq_len) - position indices
        
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Sum embeddings (논문의 E = Token + Segment + Position)
        embeddings = token_embeds + segment_embeds + position_embeds
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERT(nn.Module):
    """
    BERT Model Implementation
    
    논문 설정 (BERT-Base):
    - L = 12 (layers)
    - H = 768 (hidden size)
    - A = 12 (attention heads)
    - d_ff = 3072 (4 * H)
    - vocab_size = 30,000 (WordPiece)
    - max_seq_len = 512
    
    논문 설정 (BERT-Large):
    - L = 24
    - H = 1024
    - A = 16
    - d_ff = 4096
    """
    def __init__(
        self,
        vocab_size: int = 30000,
        max_seq_len: int = 512,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # BERT Embeddings (Token + Segment + Position)
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_seq_len, dropout)
        
        # Transformer Encoder Layers (양방향)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooler for [CLS] token (classification tasks용)
        self.pooler = nn.Linear(d_model, d_model)
        
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화 (논문의 truncated normal distribution)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_attention_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Padding mask 생성 (BERT는 양방향이므로 causal mask 없음)
        """
        # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices
            segment_ids: (batch_size, seq_len) - segment indices
            attention_mask: (batch_size, 1, 1, seq_len) - attention mask
        
        Returns:
            sequence_output: (batch_size, seq_len, d_model) - all token representations
            pooled_output: (batch_size, d_model) - [CLS] token representation
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Get embeddings (Token + Segment + Position)
        x = self.embeddings(input_ids, segment_ids)
        
        # Pass through encoder layers (양방향 attention)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        sequence_output = x  # (batch_size, seq_len, d_model)
        
        # Pool [CLS] token for classification
        # [CLS]는 항상 첫 번째 토큰
        cls_token = sequence_output[:, 0, :]  # (batch_size, d_model)
        pooled_output = torch.tanh(self.pooler(cls_token))
        
        return sequence_output, pooled_output


class BERTForPreTraining(nn.Module):
    """
    BERT Pre-training with two objectives:
    1. Masked Language Model (MLM)
    2. Next Sentence Prediction (NSP)
    
    논문 수식:
    L = L_MLM + L_NSP
    """
    def __init__(self, bert: BERT):
        super().__init__()
        self.bert = bert
        
        # MLM Head: predict masked tokens
        self.mlm_head = nn.Linear(bert.d_model, bert.embeddings.token_embedding.num_embeddings)
        self.mlm_head.weight = bert.embeddings.token_embedding.weight  # Weight tying
        
        # NSP Head: binary classification (IsNext vs NotNext)
        self.nsp_head = nn.Linear(bert.d_model, 2)
        
        # Layer norm and activation for MLM
        self.mlm_layer_norm = nn.LayerNorm(bert.d_model)
        self.mlm_dense = nn.Linear(bert.d_model, bert.d_model)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked_lm_labels: Optional[torch.Tensor] = None,
        next_sentence_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
            attention_mask: (batch_size, 1, 1, seq_len)
            masked_lm_labels: (batch_size, seq_len) - -100 for non-masked tokens
            next_sentence_labels: (batch_size,) - 0 for IsNext, 1 for NotNext
        
        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
            nsp_logits: (batch_size, 2)
            mlm_loss: scalar (if labels provided)
            nsp_loss: scalar (if labels provided)
        """
        # Forward through BERT
        sequence_output, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        
        # 1. Masked Language Model (MLM)
        # Transform hidden states for MLM prediction
        mlm_hidden = self.mlm_dense(sequence_output)
        mlm_hidden = F.gelu(mlm_hidden)
        mlm_hidden = self.mlm_layer_norm(mlm_hidden)
        mlm_logits = self.mlm_head(mlm_hidden)  # (batch_size, seq_len, vocab_size)
        
        # Calculate MLM loss if labels provided
        mlm_loss = None
        if masked_lm_labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                masked_lm_labels.view(-1),
                ignore_index=-100
            )
        
        # 2. Next Sentence Prediction (NSP)
        nsp_logits = self.nsp_head(pooled_output)  # (batch_size, 2)
        
        # Calculate NSP loss if labels provided
        nsp_loss = None
        if next_sentence_labels is not None:
            nsp_loss = F.cross_entropy(nsp_logits, next_sentence_labels)
        
        return mlm_logits, nsp_logits, mlm_loss, nsp_loss


class MaskedLanguageModelDataPreparation:
    """
    Masked Language Model (MLM) 데이터 준비
    
    논문의 masking strategy:
    - 15%의 토큰을 랜덤하게 선택
    - 선택된 토큰 중:
      * 80%는 [MASK] 토큰으로 대체
      * 10%는 랜덤 토큰으로 대체
      * 10%는 원본 유지
    """
    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        pad_token_id: int,
        cls_token_id: int,
        sep_token_id: int,
        mask_prob: float = 0.15
    ):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_prob = mask_prob
    
    def mask_tokens(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        논문의 masking strategy 구현
        
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:
            masked_input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) - -100 for non-masked tokens
        """
        labels = input_ids.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        
        # Don't mask special tokens [CLS], [SEP], [PAD]
        special_tokens_mask = (
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id) |
            (input_ids == self.pad_token_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Randomly select 15% of tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100
        
        # 80% of the time: replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10% of the time: replace with random token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% of the time: keep original token (do nothing)
        
        return input_ids, labels


class NextSentencePredictionDataPreparation:
    """
    Next Sentence Prediction (NSP) 데이터 준비
    
    논문의 NSP strategy:
    - 50%는 실제 다음 문장 (IsNext, label=0)
    - 50%는 랜덤 문장 (NotNext, label=1)
    """
    @staticmethod
    def create_nsp_data(
        sentence_a: str,
        sentence_b: str,
        is_next: bool
    ) -> Tuple[str, str, int]:
        """
        NSP 데이터 생성
        
        Args:
            sentence_a: 첫 번째 문장
            sentence_b: 두 번째 문장
            is_next: True if sentence_b follows sentence_a
        
        Returns:
            sentence_a, sentence_b, label (0 for IsNext, 1 for NotNext)
        """
        label = 0 if is_next else 1
        return sentence_a, sentence_b, label


class BERTPreTraining:
    """
    BERT Pre-training
    
    논문 설정:
    - Optimizer: Adam (β1=0.9, β2=0.999)
    - Learning rate: 1e-4
    - Learning rate warmup: first 10,000 steps
    - Batch size: 256 sequences
    - Max sequence length: 512
    - Training steps: 1,000,000
    - Weight decay: 0.01
    """
    def __init__(
        self,
        model: BERTForPreTraining,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        betas: Tuple[float, float] = (0.9, 0.999)
    ):
        self.model = model
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
        
        self.current_step = 0
    
    def get_lr(self) -> float:
        """
        Learning rate schedule with warmup
        논문: linear warmup over first 10,000 steps
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Linear decay
            return self.base_lr
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_lm_labels: torch.Tensor,
        next_sentence_labels: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        단일 training step
        
        Returns:
            total_loss, mlm_loss, nsp_loss
        """
        self.model.train()
        self.current_step += 1
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        mlm_logits, nsp_logits, mlm_loss, nsp_loss = self.model(
            input_ids,
            segment_ids,
            attention_mask,
            masked_lm_labels,
            next_sentence_labels
        )
        
        # Total loss: L = L_MLM + L_NSP
        total_loss = mlm_loss + nsp_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return total_loss.item(), mlm_loss.item(), nsp_loss.item()


class BERTForSequenceClassification(nn.Module):
    """
    BERT Fine-tuning for Sequence Classification
    
    논문에서 다루는 downstream tasks:
    - GLUE benchmark
    - SQuAD (Question Answering)
    - NER (Named Entity Recognition)
    등
    """
    def __init__(self, bert: BERT, num_labels: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.d_model, num_labels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
            attention_mask: (batch_size, 1, 1, seq_len)
            labels: (batch_size,) - classification labels
        
        Returns:
            logits: (batch_size, num_labels)
            loss: scalar (if labels provided)
        """
        # Forward through BERT
        _, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss


class BERTFineTuning:
    """
    BERT Fine-tuning
    
    논문 설정:
    - Batch size: 16 or 32
    - Learning rate: 5e-5, 3e-5, 2e-5 (task-dependent)
    - Epochs: 2-4
    - Warmup: 10% of training steps
    """
    def __init__(
        self,
        model: BERTForSequenceClassification,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1
    ):
        self.model = model
        self.warmup_ratio = warmup_ratio
        self.base_lr = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.current_step = 0
        self.total_steps = 0
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Fine-tuning step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, loss = self.model(input_ids, segment_ids, attention_mask, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        self.current_step += 1
        
        return loss.item()


# ===== 사용 예시 =====
def example_usage():
    """BERT 모델 사용 예시"""
    
    # 논문의 BERT-Base 하이퍼파라미터로 모델 생성
    bert = BERT(
        vocab_size=30000,    # WordPiece vocab
        max_seq_len=512,     # max sequence length
        d_model=768,         # hidden size (H)
        num_layers=12,       # number of layers (L)
        num_heads=12,        # number of attention heads (A)
        d_ff=3072,           # FFN dimension (4*H)
        dropout=0.1
    )
    
    print(f"BERT-Base Parameters: {sum(p.numel() for p in bert.parameters()):,}")
    print(f"논문의 설정: ~110M parameters")
    
    # 1. Pre-training 예시
    print("\n=== Pre-training Example ===")
    pretrain_model = BERTForPreTraining(bert)
    pretrainer = BERTPreTraining(pretrain_model, learning_rate=1e-4)
    
    # Dummy data
    batch_size = 4
    seq_len = 128
    vocab_size = 30000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    segment_ids[:, seq_len//2:] = 1  # Second half is sentence B
    attention_mask = bert.create_attention_mask(input_ids)
    
    # MLM labels (masked tokens)
    masked_lm_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    masked_positions = torch.randint(0, seq_len, (batch_size, 10))
    for i in range(batch_size):
        masked_lm_labels[i, masked_positions[i]] = input_ids[i, masked_positions[i]]
    
    # NSP labels
    next_sentence_labels = torch.randint(0, 2, (batch_size,))
    
    total_loss, mlm_loss, nsp_loss = pretrainer.train_step(
        input_ids, segment_ids, attention_mask, masked_lm_labels, next_sentence_labels
    )
    print(f"Total Loss: {total_loss:.4f}")
    print(f"MLM Loss: {mlm_loss:.4f}")
    print(f"NSP Loss: {nsp_loss:.4f}")
    
    # 2. Masking Strategy 예시
    print("\n=== Masking Strategy Example ===")
    masker = MaskedLanguageModelDataPreparation(
        vocab_size=30000,
        mask_token_id=103,  # [MASK] token
        pad_token_id=0,
        cls_token_id=101,   # [CLS] token
        sep_token_id=102    # [SEP] token
    )
    
    original_ids = torch.randint(104, vocab_size, (2, 20))
    original_ids[:, 0] = 101  # [CLS]
    original_ids[:, -1] = 102  # [SEP]
    
    masked_ids, labels = masker.mask_tokens(original_ids)
    print(f"Original tokens: {original_ids[0, :10]}")
    print(f"Masked tokens: {masked_ids[0, :10]}")
    print(f"Labels: {labels[0, :10]}")
    
    # 3. Fine-tuning 예시 (Sequence Classification)
    print("\n=== Fine-tuning Example ===")
    num_classes = 2  # binary classification
    classifier = BERTForSequenceClassification(bert, num_labels=num_classes)
    finetuner = BERTFineTuning(classifier, learning_rate=3e-5)
    
    labels = torch.randint(0, num_classes, (batch_size,))
    loss = finetuner.train_step(input_ids, segment_ids, attention_mask, labels)
    print(f"Fine-tuning Loss: {loss:.4f}")
    
    # 4. BERT vs GPT 비교
    print("\n=== BERT vs GPT ===")
    print("BERT:")
    print("  - Bidirectional Transformer ENCODER")
    print("  - Pre-training: MLM + NSP")
    print("  - Can see all tokens (양방향)")
    print("  - Best for: Understanding tasks (classification, NER, QA)")
    print("\nGPT:")
    print("  - Unidirectional Transformer DECODER")
    print("  - Pre-training: Language Modeling (left-to-right)")
    print("  - Can only see previous tokens (단방향)")
    print("  - Best for: Generation tasks")


if __name__ == "__main__":
    """
    BERT 논문 구현 요약:
    
    1. 아키텍처 (BERT-Base):
       - L=12 Transformer Encoder layers
       - H=768 hidden size
       - A=12 attention heads
       - d_ff=3072 feed-forward dimension
       - Bidirectional self-attention (양방향)
    
    2. Pre-training Objectives:
       - MLM (Masked Language Model): 15% random masking
         * 80% [MASK], 10% random token, 10% unchanged
       - NSP (Next Sentence Prediction): 50% IsNext, 50% NotNext
       - Total Loss: L = L_MLM + L_NSP
    
    3. Embeddings:
       - Token Embeddings (WordPiece, 30K vocab)
       - Segment Embeddings (sentence A/B)
       - Position Embeddings (learned, not sinusoidal)
       - E = Token + Segment + Position
    
    4. Pre-training Settings:
       - Corpus: BooksCorpus (800M words) + English Wikipedia (2,500M words)
       - Batch size: 256 sequences
       - Steps: 1,000,000
       - Learning rate: 1e-4 with warmup (10K steps)
       - Optimizer: Adam (β1=0.9, β2=0.999)
    
    5. Fine-tuning Settings:
       - Batch size: 16 or 32
       - Learning rate: 2e-5, 3e-5, 5e-5
       - Epochs: 2-4
       - Task-specific layers added on top
    
    6. Key Innovations:
       - Bidirectional pre-training (vs GPT's left-to-right)
       - Masked Language Model (novel pre-training task)
       - Next Sentence Prediction (sentence-level understanding)
       - Transfer learning with minimal task-specific parameters
    """
    
    print("=" * 70)
    print("BERT Implementation")
    print("BERT: Pre-training of Deep Bidirectional Transformers")
    print("for Language Understanding")
    print("=" * 70)
    
    example_usage()
    
    print("\n" + "=" * 70)
    print("구현 완료!")
    print("=" * 70)
