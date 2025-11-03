"""
GPT-1 Implementation based on "Improving Language Understanding by Generative Pre-Training"
논문의 핵심 수식과 개념을 충실히 구현

주요 구성 요소:
1. Transformer Decoder 아키텍처
2. Multi-Head Self-Attention
3. Position-wise Feed-Forward Networks
4. Unsupervised Pre-training (Language Modeling Objective)
5. Supervised Fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 메커니즘
    논문 수식: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 head의 dimension
        
        # Q, K, V projection matrices (논문의 W^Q, W^K, W^V)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection (논문의 W^O)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        논문 수식: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        """
        # Q: (batch_size, num_heads, seq_len, d_k)
        # K: (batch_size, num_heads, seq_len, d_k)
        # V: (batch_size, num_heads, seq_len, d_k)
        
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask 적용 (GPT는 auto-regressive이므로 미래 토큰을 볼 수 없음)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax(QK^T / sqrt(d_k))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # softmax(...)V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len) - causal mask
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections: QW^Q, KW^K, VW^V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concat heads: (batch_size, num_heads, seq_len, d_k) 
        # -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection: Concat(...)W^O
        output = self.W_o(attn_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    논문 수식: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        # max(0, ...) is ReLU activation
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block
    논문 구조:
    1. Masked Multi-Head Attention + Residual Connection + Layer Norm
    2. Position-wise FFN + Residual Connection + Layer Norm
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: causal mask
        """
        # Masked Multi-Head Attention + Residual + LayerNorm
        attn_output = self.attention(x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # Feed-Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_output))
        
        return x


class GPT1(nn.Module):
    """
    GPT-1 Model Implementation
    논문 설정:
    - 12 layers (Transformer blocks)
    - d_model = 768 (hidden size)
    - num_heads = 12
    - d_ff = 3072 (4 * d_model)
    - vocab_size = 40,000 (BPE)
    - max_seq_len = 512
    - dropout = 0.1
    """
    def __init__(
        self,
        vocab_size: int = 40000,
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
        
        # Token Embedding (논문의 W_e)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding (논문의 W_p)
        # GPT-1은 learned positional embeddings 사용
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Blocks (12 layers)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (언어 모델링 head)
        self.ln_f = nn.LayerNorm(d_model)  # Final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (token embedding과 output layer 공유)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화 (논문의 N(0, 0.02))"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask 생성 (auto-regressive를 위한 mask)
        Upper triangular matrix로 미래 토큰을 masking
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices
            labels: (batch_size, seq_len) - target tokens for language modeling
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        
        # Embeddings: h_0 = U*W_e + W_p (논문 수식)
        token_embeds = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        pos_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Transformer blocks: h_l = transformer_block(h_{l-1})
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # 논문의 L1(U) = Σ log P(u_i | u_{i-k}, ..., u_{i-1}; Θ)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Auto-regressive text generation
        
        Args:
            input_ids: (batch_size, seq_len) - starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if sequence is too long
                idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else \
                           input_ids[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self.forward(idx_cond)
                
                # Get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


class GPT1PreTraining:
    """
    GPT-1 Pre-training (Unsupervised Learning)
    논문의 Objective: L1(U) = Σ log P(u_i | u_{i-k}, ..., u_{i-1}; Θ)
    """
    def __init__(
        self,
        model: GPT1,
        learning_rate: float = 2.5e-4,  # 논문 설정
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999)
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
        
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        단일 training step
        논문 수식: minimize -Σ log P(u_i | context)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, loss = self.model(input_ids, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (논문에서는 global norm으로 clip)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()


class GPT1FineTuning:
    """
    GPT-1 Fine-tuning (Supervised Learning)
    논문의 Objective: L2(C) = Σ log P(y | x^1, ..., x^m)
    Combined: L3(C) = L2(C) + λ * L1(C)
    """
    def __init__(
        self,
        model: GPT1,
        num_labels: int,
        learning_rate: float = 6.25e-5,  # 논문에서 fine-tuning lr
        weight_decay: float = 0.01,
        lambda_lm: float = 0.5  # 논문의 λ (auxiliary LM objective weight)
    ):
        self.model = model
        self.lambda_lm = lambda_lm
        
        # Classification head (task-specific)
        self.classifier = nn.Linear(model.d_model, num_labels)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # Optimizer for both model and classifier
        params = list(model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        task_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Fine-tuning step with auxiliary language modeling objective
        
        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) - for LM loss
            task_labels: (batch_size,) - for classification loss
        
        Returns:
            total_loss, task_loss
        """
        self.model.train()
        self.classifier.train()
        self.optimizer.zero_grad()
        
        # Forward pass through GPT
        logits, lm_loss = self.model(input_ids, labels)
        
        # Get last hidden state for classification
        # Extract [CLS] token representation (last token)
        hidden_states = logits[:, -1, :]  # (batch_size, vocab_size)
        
        # We need actual hidden states, not logits
        # Re-compute without lm_head
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        
        token_embeds = self.model.token_embedding(input_ids)
        pos_embeds = self.model.position_embedding(position_ids)
        x = self.model.dropout(token_embeds + pos_embeds)
        
        mask = self.model.create_causal_mask(seq_len, device)
        for block in self.model.transformer_blocks:
            x = block(x, mask)
        x = self.model.ln_f(x)
        
        # Classification on last token
        cls_hidden = x[:, -1, :]  # (batch_size, d_model)
        task_logits = self.classifier(cls_hidden)
        
        # Task loss: L2(C) = Σ log P(y | x)
        task_loss = F.cross_entropy(task_logits, task_labels)
        
        # Combined loss: L3(C) = L2(C) + λ * L1(C)
        if lm_loss is not None:
            total_loss = task_loss + self.lambda_lm * lm_loss
        else:
            total_loss = task_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return total_loss.item(), task_loss.item()


# ===== 사용 예시 =====
def example_usage():
    """GPT-1 모델 사용 예시"""
    
    # 논문의 하이퍼파라미터로 모델 생성
    model = GPT1(
        vocab_size=40000,    # BPE vocab size
        max_seq_len=512,     # max context length
        d_model=768,         # hidden dimension
        num_layers=12,       # transformer layers
        num_heads=12,        # attention heads
        d_ff=3072,           # FFN dimension (4 * d_model)
        dropout=0.1          # dropout rate
    )
    
    print(f"GPT-1 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"논문의 설정: ~117M parameters")
    
    # 1. Pre-training 예시
    print("\n=== Pre-training Example ===")
    pretrainer = GPT1PreTraining(model, learning_rate=2.5e-4)
    
    # Dummy data (실제로는 large corpus 필요)
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 40000, (batch_size, seq_len))
    labels = torch.randint(0, 40000, (batch_size, seq_len))
    
    loss = pretrainer.train_step(input_ids, labels)
    print(f"Pre-training loss: {loss:.4f}")
    
    # 2. Fine-tuning 예시 (분류 task)
    print("\n=== Fine-tuning Example ===")
    num_classes = 2  # binary classification
    finetuner = GPT1FineTuning(model, num_labels=num_classes, learning_rate=6.25e-5)
    
    task_labels = torch.randint(0, num_classes, (batch_size,))
    total_loss, task_loss = finetuner.train_step(input_ids, labels, task_labels)
    print(f"Fine-tuning total loss: {total_loss:.4f}")
    print(f"Fine-tuning task loss: {task_loss:.4f}")
    
    # 3. Text Generation 예시
    print("\n=== Text Generation Example ===")
    model.eval()
    start_tokens = torch.randint(0, 40000, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Original length: 10, Generated length: {generated.shape[1]}")


if __name__ == "__main__":
    """
    GPT-1 논문 구현 요약:
    
    1. 아키텍처:
       - Transformer Decoder (12 layers)
       - Multi-Head Attention (12 heads, d_k = 64)
       - Position-wise FFN (d_ff = 3072)
       - Learned positional embeddings
       - Layer normalization & residual connections
    
    2. Pre-training:
       - Objective: L1(U) = Σ log P(u_i | u_{i-k}, ..., u_{i-1})
       - Dataset: BooksCorpus (7,000 unique books)
       - Context size: 512 tokens
       - Batch size: 64
       - Learning rate: 2.5e-4
       - Training: 100 epochs
    
    3. Fine-tuning:
       - Objective: L3(C) = L2(C) + λ * L1(C)
       - λ = 0.5 (auxiliary LM objective weight)
       - Learning rate: 6.25e-5
       - Batch size: 32
       - Training: 3 epochs
       - Linear learning rate decay
    
    4. 주요 수식:
       - Attention: softmax(QK^T / sqrt(d_k))V
       - FFN: max(0, xW_1 + b_1)W_2 + b_2
       - h_l = transformer_block(h_{l-1})
    """
    
    print("=" * 60)
    print("GPT-1 Implementation (Improving Language Understanding by Generative Pre-Training)")
    print("=" * 60)
    
    example_usage()
    
    print("\n" + "=" * 60)
    print("구현 완료!")
    print("=" * 60)
