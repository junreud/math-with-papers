# GPT-1 Implementation

## 논문 정보
**제목**: Improving Language Understanding by Generative Pre-Training  
**저자**: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever  
**기관**: OpenAI  
**년도**: 2018

---

## 핵심 아이디어

GPT-1은 **Generative Pre-Training**을 통해 unlabeled text에서 언어 표현을 학습한 후, 특정 task에 대해 discriminative fine-tuning을 수행하는 semi-supervised 접근법입니다.

### 주요 특징
1. **Transformer Decoder 아키텍처** 사용
2. **단방향 (Left-to-right) Language Modeling**
3. **Unsupervised Pre-training + Supervised Fine-tuning**
4. **Minimal task-specific architecture modification**

---

## 아키텍처 구현

### 1. Multi-Head Attention

**논문 수식**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**코드 구현** (`MultiHeadAttention` 클래스):
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        # d_model = 768, num_heads = 12
        # d_k = d_model // num_heads = 64
        
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
```

**핵심 구현**:
- `scaled_dot_product_attention()`: QK^T / sqrt(d_k) 계산 → softmax → V와 곱셈
- **Causal Mask**: Auto-regressive를 위해 미래 토큰 masking (upper triangular)
- Multi-head로 split → attention → concat → output projection

---

### 2. Position-wise Feed-Forward Network

**논문 수식**:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**코드 구현** (`PositionwiseFeedForward` 클래스):
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        self.W_1 = nn.Linear(d_model, d_ff)   # 768 → 3072
        self.W_2 = nn.Linear(d_ff, d_model)   # 3072 → 768
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(self.dropout(F.relu(self.W_1(x))))
```

**특징**:
- Two linear transformations with ReLU activation
- d_ff = 4 * d_model = 3072
- Position마다 독립적으로 적용

---

### 3. Transformer Block

**논문 구조**:
```
h_l = transformer_block(h_{l-1})
    = LayerNorm(h_{l-1} + MultiHeadAttention(h_{l-1}))
    = LayerNorm(h + FFN(h))
```

**코드 구현** (`TransformerBlock` 클래스):
```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask):
        # 1. Masked Multi-Head Attention + Residual + LayerNorm
        attn_output = self.attention(x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_output))
        
        return x
```

**구성 요소**:
- Masked Multi-Head Self-Attention
- Residual Connection
- Layer Normalization
- Position-wise FFN

---

### 4. GPT-1 Main Model

**논문 하이퍼파라미터** (117M parameters):
- **Layers (L)**: 12
- **Hidden Size (d_model)**: 768
- **Attention Heads**: 12
- **FFN Dimension (d_ff)**: 3072 (4 * 768)
- **Vocabulary**: 40,000 (BPE)
- **Max Sequence Length**: 512
- **Dropout**: 0.1

**코드 구현** (`GPT1` 클래스):
```python
class GPT1(nn.Module):
    def __init__(self, vocab_size=40000, max_seq_len=512, d_model=768, 
                 num_layers=12, num_heads=12, d_ff=3072, dropout=0.1):
        
        # Token Embedding (논문의 W_e)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding (논문의 W_p) - LEARNED embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 12 Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer (Language Modeling Head)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (논문에서 사용)
        self.lm_head.weight = self.token_embedding.weight
```

**Embedding 계산**:
```
h_0 = U*W_e + W_p
```
- U: input token indices
- W_e: token embedding matrix
- W_p: position embedding matrix

---

## Pre-training

### Objective Function

**논문 수식**:
```
L1(U) = Σ log P(u_i | u_{i-k}, ..., u_{i-1}; Θ)
```

**의미**: k 토큰 context를 이용해 다음 토큰 예측 (Language Modeling)

**코드 구현** (`GPT1PreTraining` 클래스):
```python
class GPT1PreTraining:
    def train_step(self, input_ids, labels):
        # Forward pass
        logits, loss = self.model(input_ids, labels)
        
        # Loss calculation (논문의 L1)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
```

### Pre-training 설정

**논문 명시**:
- **Dataset**: BooksCorpus (7,000 unique books, ~5GB text)
- **Context Size**: 512 tokens
- **Batch Size**: 64
- **Learning Rate**: 2.5e-4
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Epochs**: 100
- **Learning Rate Schedule**: Cosine annealing

**코드 구현**:
```python
pretrainer = GPT1PreTraining(
    model,
    learning_rate=2.5e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

---

## Fine-tuning

### Objective Function

**논문 수식**:
```
L2(C) = Σ log P(y | x^1, ..., x^m)          # Task-specific objective
L3(C) = L2(C) + λ * L1(C)                   # Combined objective
```

**의미**:
- L2: Supervised task loss (classification, QA, etc.)
- L1: Auxiliary language modeling loss
- λ = 0.5: Weight for auxiliary objective

**코드 구현** (`GPT1FineTuning` 클래스):
```python
class GPT1FineTuning:
    def train_step(self, input_ids, labels, task_labels):
        # Language Modeling Loss (L1)
        logits, lm_loss = self.model(input_ids, labels)
        
        # Task-specific Loss (L2)
        cls_hidden = x[:, -1, :]  # Last token representation
        task_logits = self.classifier(cls_hidden)
        task_loss = F.cross_entropy(task_logits, task_labels)
        
        # Combined Loss (L3)
        total_loss = task_loss + self.lambda_lm * lm_loss
        
        total_loss.backward()
        self.optimizer.step()
```

### Fine-tuning 설정

**논문 명시**:
- **Batch Size**: 32
- **Learning Rate**: 6.25e-5
- **Epochs**: 3
- **Learning Rate Schedule**: Linear decay to 0
- **λ (lambda)**: 0.5

**Task-Specific Input Transformations**:
1. **Classification**: [Start] Text [Extract]
2. **Entailment**: [Start] Premise [Delim] Hypothesis [Extract]
3. **Similarity**: [Start] Text1 [Delim] Text2 [Extract]
4. **Multiple Choice**: [Start] Context [Delim] Answer_i [Extract] (각 선택지마다)

---

## 핵심 구현 디테일

### 1. Causal Mask

**목적**: Auto-regressive generation을 위해 미래 토큰 masking

**코드**:
```python
def create_causal_mask(self, seq_len, device):
    # Lower triangular matrix (미래 토큰은 0으로 masking)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

**적용**:
```python
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

### 2. Weight Initialization

**논문 명시**: N(0, 0.02)

**코드**:
```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

---

### 3. Text Generation

**Auto-regressive generation**:

**코드** (`generate()` 메서드):
```python
def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # Forward pass
        logits, _ = self.forward(input_ids)
        
        # Get last token logits
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids
```

---

## 논문 실험 결과

### Natural Language Inference
- **MultiNLI**: 82.1% (이전 SOTA 대비 +4.5%)
- **SNLI**: 89.9%

### Question Answering
- **RACE**: 59.0% (이전 SOTA 대비 +5.7%)

### Semantic Similarity
- **STS-B**: Pearson correlation 82.0

### Classification
- **SST-2**: 91.3%

---

## GPT-1 vs BERT 비교

| 특징 | GPT-1 | BERT |
|------|-------|------|
| **아키텍처** | Transformer Decoder | Transformer Encoder |
| **방향성** | 단방향 (Left-to-right) | 양방향 (Bidirectional) |
| **Pre-training** | Language Modeling (LM) | MLM + NSP |
| **Mask** | Causal Mask (미래 숨김) | Padding Mask만 |
| **강점** | Text Generation | Text Understanding |
| **Attention** | 이전 토큰만 참조 | 모든 토큰 참조 |
| **Embedding** | Token + Position | Token + Segment + Position |

---

## 코드 실행 예시

```python
# 1. 모델 생성 (논문 설정)
model = GPT1(
    vocab_size=40000,
    max_seq_len=512,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    dropout=0.1
)

# 2. Pre-training
pretrainer = GPT1PreTraining(model, learning_rate=2.5e-4)
loss = pretrainer.train_step(input_ids, labels)

# 3. Fine-tuning
finetuner = GPT1FineTuning(model, num_labels=2, learning_rate=6.25e-5)
total_loss, task_loss = finetuner.train_step(input_ids, labels, task_labels)

# 4. Text Generation
generated = model.generate(start_tokens, max_new_tokens=50, temperature=0.8, top_k=50)
```

---

## 주요 수식 요약

### 1. Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

### 2. Feed-Forward
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### 3. Pre-training Objective
```
L1(U) = Σ log P(u_i | u_{i-k}, ..., u_{i-1}; Θ)
```

### 4. Fine-tuning Objective
```
L3(C) = L2(C) + λ * L1(C)
```

### 5. Transformer Layer
```
h_l = transformer_block(h_{l-1})
h_0 = U*W_e + W_p
```

---

## 구현 완성도

✅ Multi-Head Attention with Causal Mask  
✅ Position-wise Feed-Forward Network  
✅ 12-layer Transformer Decoder  
✅ Token + Position Embeddings  
✅ Pre-training with Language Modeling  
✅ Fine-tuning with Auxiliary LM Loss  
✅ Text Generation (Auto-regressive)  
✅ Weight Tying  
✅ Gradient Clipping  
✅ 논문 하이퍼파라미터 정확히 반영  

---

## 참고사항

- 실제 학습을 위해서는 BPE tokenizer 구현 필요
- BooksCorpus 데이터셋 준비 필요
- 실제 논문에서는 분산 학습 사용 (8 GPUs)
- Fine-tuning시 task-specific input format 변환 필요

**논문 링크**: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
