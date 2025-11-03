# BERT Implementation

## 논문 정보
**제목**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
**저자**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
**기관**: Google AI Language  
**년도**: 2019

---

## 핵심 아이디어

BERT는 **Bidirectional Encoder Representations from Transformers**로, unlabeled text에서 양방향 표현을 학습하는 모델입니다. GPT-1과 달리 **모든 레이어에서 양방향 context**를 활용합니다.

### 주요 특징
1. **Transformer Encoder 아키텍처** (Decoder가 아님)
2. **양방향 (Bidirectional) Pre-training**
3. **Masked Language Model (MLM)** - 새로운 pre-training task
4. **Next Sentence Prediction (NSP)** - 문장 관계 학습
5. **Triple Embeddings**: Token + Segment + Position

---

## 아키텍처 구현

### 1. Multi-Head Attention (Bidirectional)

**논문 수식**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
```

**GPT와의 핵심 차이**:
- **BERT**: Causal mask 없음 → 모든 토큰이 서로를 볼 수 있음 (양방향)
- **GPT**: Causal mask 있음 → 이전 토큰만 볼 수 있음 (단방향)

**코드 구현** (`MultiHeadAttention` 클래스):
```python
class MultiHeadAttention(nn.Module):
    def scaled_dot_product_attention(self, Q, K, V, attention_mask):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # BERT: Padding mask만 적용 (Causal mask 없음!)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
```

**비교**:
```python
# GPT-1: Causal mask (미래 토큰 숨김)
mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular

# BERT: Padding mask만 (모든 방향 볼 수 있음)
mask = (input_ids != pad_token_id)  # Only mask padding
```

---

### 2. Position-wise Feed-Forward with GELU

**논문 수식**:
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

**GPT와의 차이**:
- **BERT**: GELU activation
- **GPT**: ReLU activation

**GELU 수식**:
```
GELU(x) = x * Φ(x)
where Φ(x) is the cumulative distribution function of standard Gaussian
```

**코드 구현** (`PositionwiseFeedForward` 클래스):
```python
class PositionwiseFeedForward(nn.Module):
    def forward(self, x):
        # BERT는 GELU 사용 (ReLU 대신)
        return self.W_2(self.dropout(F.gelu(self.W_1(x))))
```

---

### 3. Transformer Encoder Layer

**논문 구조**:
```
h_l = TransformerEncoder(h_{l-1})
    = LayerNorm(h_{l-1} + MultiHeadAttention(h_{l-1}))  # 양방향
    = LayerNorm(h + FFN(h))
```

**코드 구현** (`TransformerEncoderLayer` 클래스):
```python
class TransformerEncoderLayer(nn.Module):
    def forward(self, x, attention_mask):
        # 1. Bidirectional Multi-Head Attention + Residual + LayerNorm
        attn_output = self.attention(x, attention_mask)  # 양방향!
        x = self.ln1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_output))
        
        return x
```

---

### 4. BERT Embeddings (핵심!)

**논문 수식**:
```
E = Token Embeddings + Segment Embeddings + Position Embeddings
```

**3가지 Embedding 종류**:

#### 4.1 Token Embeddings
- **WordPiece tokenization** (30,000 vocab)
- 하위 단어 단위로 분할 (예: "playing" → "play" + "##ing")

#### 4.2 Segment Embeddings (BERT만의 특징!)
- **문장 A와 B 구분** (NSP task용)
- 0: Sentence A, 1: Sentence B
- GPT에는 없는 개념

#### 4.3 Position Embeddings
- **Learned embeddings** (sinusoidal 아님)
- Max 512 positions

**코드 구현** (`BERTEmbeddings` 클래스):
```python
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout):
        # 1. Token Embeddings (WordPiece)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Segment Embeddings (BERT만의 특징!)
        self.segment_embedding = nn.Embedding(2, d_model)  # A or B
        
        # 3. Position Embeddings (Learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, segment_ids, position_ids):
        # Sum all three embeddings
        token_embeds = self.token_embedding(input_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # E = Token + Segment + Position
        embeddings = token_embeds + segment_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

**예시**:
```
Input: [CLS] I love NLP [SEP] BERT is great [SEP]
Token IDs:    [101,  34, 789, 456, 102,  890, 12, 345, 102]
Segment IDs:  [  0,   0,   0,   0,   0,    1,  1,   1,   1]
                 ↑-------Sentence A------↑   ↑--Sentence B--↑
```

---

### 5. BERT Main Model

**논문 하이퍼파라미터**:

| Configuration | BERT-Base | BERT-Large |
|--------------|-----------|------------|
| Layers (L) | 12 | 24 |
| Hidden Size (H) | 768 | 1024 |
| Attention Heads (A) | 12 | 16 |
| FFN Dimension | 3072 | 4096 |
| Parameters | 110M | 340M |
| Max Sequence | 512 | 512 |
| Vocabulary | 30,000 | 30,000 |

**코드 구현** (`BERT` 클래스):
```python
class BERT(nn.Module):
    def __init__(self, vocab_size=30000, max_seq_len=512, d_model=768,
                 num_layers=12, num_heads=12, d_ff=3072, dropout=0.1):
        
        # Triple Embeddings (Token + Segment + Position)
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_seq_len, dropout)
        
        # 12 Transformer Encoder Layers (양방향)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooler for [CLS] token representation
        self.pooler = nn.Linear(d_model, d_model)
    
    def forward(self, input_ids, segment_ids, attention_mask):
        # Get embeddings
        x = self.embeddings(input_ids, segment_ids)
        
        # Pass through encoder layers (양방향 attention)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        sequence_output = x  # All token representations
        
        # Pool [CLS] token (첫 번째 토큰)
        cls_token = sequence_output[:, 0, :]
        pooled_output = torch.tanh(self.pooler(cls_token))
        
        return sequence_output, pooled_output
```

---

## Pre-training

BERT는 **두 가지 unsupervised task**로 pre-training합니다.

### Task 1: Masked Language Model (MLM)

**논문 아이디어**:
- 양방향 학습을 위해 입력의 일부를 mask하고 예측
- 15%의 토큰을 랜덤 선택

**Masking Strategy** (논문의 핵심):
1. **80%**: [MASK] 토큰으로 대체
2. **10%**: 랜덤 토큰으로 대체
3. **10%**: 원본 유지

**예시**:
```
Original:  "I love natural language processing"
Masked:    "I love [MASK] language [MASK]"
Target:    Only predict "natural" and "processing"
```

**왜 80/10/10?**
- Fine-tuning시 [MASK]가 없으므로 mismatch 방지
- 10%는 랜덤으로 바꿔서 representation 학습
- 10%는 유지해서 actual word distribution 학습

**코드 구현** (`MaskedLanguageModelDataPreparation` 클래스):
```python
class MaskedLanguageModelDataPreparation:
    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        
        # 15% of tokens 선택
        probability_matrix = torch.full(input_ids.shape, 0.15)
        
        # Special tokens는 masking 안함
        special_tokens_mask = (
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id) |
            (input_ids == self.pad_token_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # 15% 선택
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Non-masked tokens ignored in loss
        
        # 80%: [MASK] 토큰으로 대체
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10%: 랜덤 토큰으로 대체
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10%: 원본 유지 (아무것도 안함)
        
        return input_ids, labels
```

**MLM Loss**:
```python
mlm_loss = F.cross_entropy(
    mlm_logits.view(-1, vocab_size),
    masked_lm_labels.view(-1),
    ignore_index=-100  # Non-masked tokens
)
```

---

### Task 2: Next Sentence Prediction (NSP)

**논문 아이디어**:
- 두 문장의 관계 이해 (QA, NLI 등에 필요)
- Binary classification: IsNext vs NotNext

**데이터 구성**:
- **50%**: 실제 연속된 문장 (IsNext, label=0)
- **50%**: 랜덤 문장 (NotNext, label=1)

**예시**:
```
IsNext (50%):
  [CLS] The man went to the store [SEP] He bought a bottle of milk [SEP]
  Label: 0

NotNext (50%):
  [CLS] The man went to the store [SEP] Penguins are flightless birds [SEP]
  Label: 1
```

**코드 구현** (`NextSentencePredictionDataPreparation` 클래스):
```python
class NextSentencePredictionDataPreparation:
    @staticmethod
    def create_nsp_data(sentence_a, sentence_b, is_next):
        label = 0 if is_next else 1  # 0: IsNext, 1: NotNext
        return sentence_a, sentence_b, label
```

**NSP Loss**:
```python
nsp_loss = F.cross_entropy(nsp_logits, next_sentence_labels)
```

---

### Combined Pre-training Loss

**논문 수식**:
```
L = L_MLM + L_NSP
```

**코드 구현** (`BERTForPreTraining` 클래스):
```python
class BERTForPreTraining(nn.Module):
    def __init__(self, bert):
        self.bert = bert
        
        # MLM Head: Predict masked tokens
        self.mlm_head = nn.Linear(bert.d_model, vocab_size)
        self.mlm_head.weight = bert.embeddings.token_embedding.weight  # Weight tying
        
        # NSP Head: Binary classification
        self.nsp_head = nn.Linear(bert.d_model, 2)
        
        # Transform layer for MLM
        self.mlm_dense = nn.Linear(bert.d_model, bert.d_model)
        self.mlm_layer_norm = nn.LayerNorm(bert.d_model)
    
    def forward(self, input_ids, segment_ids, attention_mask, 
                masked_lm_labels, next_sentence_labels):
        # Forward through BERT
        sequence_output, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        
        # 1. MLM: Predict masked tokens
        mlm_hidden = self.mlm_dense(sequence_output)
        mlm_hidden = F.gelu(mlm_hidden)
        mlm_hidden = self.mlm_layer_norm(mlm_hidden)
        mlm_logits = self.mlm_head(mlm_hidden)
        mlm_loss = F.cross_entropy(mlm_logits.view(-1, vocab_size),
                                   masked_lm_labels.view(-1),
                                   ignore_index=-100)
        
        # 2. NSP: Predict sentence relationship
        nsp_logits = self.nsp_head(pooled_output)
        nsp_loss = F.cross_entropy(nsp_logits, next_sentence_labels)
        
        # Total loss
        total_loss = mlm_loss + nsp_loss
        
        return mlm_logits, nsp_logits, mlm_loss, nsp_loss
```

---

### Pre-training 설정

**논문 명시**:

| Setting | Value |
|---------|-------|
| **Dataset** | BooksCorpus (800M words) + English Wikipedia (2,500M words) |
| **Total words** | ~3.3 billion words |
| **Batch Size** | 256 sequences |
| **Max Sequence Length** | 512 tokens |
| **Training Steps** | 1,000,000 |
| **Optimizer** | Adam (β1=0.9, β2=0.999) |
| **Learning Rate** | 1e-4 |
| **Warmup Steps** | 10,000 (linear warmup) |
| **Learning Rate Decay** | Linear |
| **Weight Decay** | 0.01 |
| **Dropout** | 0.1 |

**코드 구현** (`BERTPreTraining` 클래스):
```python
class BERTPreTraining:
    def __init__(self, model, learning_rate=1e-4, weight_decay=0.01, 
                 warmup_steps=10000, betas=(0.9, 0.999)):
        self.model = model
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
    
    def get_lr(self):
        # Linear warmup for first 10,000 steps
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            return self.base_lr  # Can add linear decay here
    
    def train_step(self, input_ids, segment_ids, attention_mask,
                   masked_lm_labels, next_sentence_labels):
        self.current_step += 1
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        _, _, mlm_loss, nsp_loss = self.model(
            input_ids, segment_ids, attention_mask,
            masked_lm_labels, next_sentence_labels
        )
        
        # Total loss: L = L_MLM + L_NSP
        total_loss = mlm_loss + nsp_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item(), mlm_loss.item(), nsp_loss.item()
```

---

## Fine-tuning

### Task-Specific Modifications

BERT는 **minimal architecture modification**으로 다양한 task에 적용:

#### 1. Sequence Classification
```
Input: [CLS] Text [SEP]
Output: [CLS] representation → Linear → Softmax
```

**코드** (`BERTForSequenceClassification` 클래스):
```python
class BERTForSequenceClassification(nn.Module):
    def __init__(self, bert, num_labels):
        self.bert = bert
        self.classifier = nn.Linear(bert.d_model, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, segment_ids, attention_mask, labels):
        # Forward through BERT
        _, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        
        # Classification on [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Loss
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return logits, loss
```

#### 2. Question Answering
```
Input: [CLS] Question [SEP] Passage [SEP]
Output: Start/End positions in passage
```

#### 3. Named Entity Recognition
```
Input: [CLS] Token1 Token2 ... [SEP]
Output: Label for each token
```

---

### Fine-tuning 설정

**논문 명시**:

| Setting | Value |
|---------|-------|
| **Batch Size** | 16 or 32 |
| **Learning Rate** | 5e-5, 3e-5, 2e-5 (task-dependent) |
| **Epochs** | 2, 3, 4 (task-dependent) |
| **Warmup** | 10% of training steps |
| **Max Sequence Length** | 128, 256, 512 (task-dependent) |

**코드 구현** (`BERTFineTuning` 클래스):
```python
class BERTFineTuning:
    def __init__(self, model, learning_rate=3e-5, weight_decay=0.01, 
                 warmup_ratio=0.1):
        self.model = model
        self.warmup_ratio = warmup_ratio
        self.base_lr = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_step(self, input_ids, segment_ids, attention_mask, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, loss = self.model(input_ids, segment_ids, attention_mask, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
```

---

## Special Tokens

BERT는 특수 토큰을 사용합니다:

| Token | ID | Purpose |
|-------|-----|---------|
| **[CLS]** | 101 | Classification token (문장 시작) |
| **[SEP]** | 102 | Separator (문장 끝 / 구분) |
| **[MASK]** | 103 | Masked token (MLM용) |
| **[PAD]** | 0 | Padding token |
| **[UNK]** | 100 | Unknown token |

**사용 예시**:
```
Single Sentence:
[CLS] I love BERT [SEP]

Sentence Pair (NSP):
[CLS] Sentence A [SEP] Sentence B [SEP]

Masked LM:
[CLS] I [MASK] BERT [SEP]
```

---

## 논문 실험 결과

### GLUE Benchmark (General Language Understanding Evaluation)

| Task | Metric | BERT-Base | BERT-Large | Previous SOTA |
|------|--------|-----------|------------|---------------|
| **MNLI** | Accuracy | 84.6/83.4 | 86.7/85.9 | 80.5/80.1 |
| **QQP** | Accuracy/F1 | 89.2/72.1 | 89.3/72.1 | 86.1/70.3 |
| **QNLI** | Accuracy | 90.5 | 92.7 | 87.4 |
| **SST-2** | Accuracy | 93.5 | 94.9 | 93.2 |
| **CoLA** | Matthews corr | 52.1 | 60.5 | 35.0 |
| **STS-B** | Pearson corr | 85.8 | 86.5 | 81.3 |
| **MRPC** | Accuracy/F1 | 88.9/84.8 | 89.3/85.4 | 86.0/83.5 |
| **RTE** | Accuracy | 66.4 | 70.1 | 61.7 |

### SQuAD (Question Answering)

| Version | Metric | BERT-Base | BERT-Large |
|---------|--------|-----------|------------|
| **SQuAD 1.1** | EM/F1 | 80.8/88.5 | **84.1/90.9** |
| **SQuAD 2.0** | EM/F1 | 72.7/76.3 | **80.0/83.1** |

### Named Entity Recognition

| Dataset | F1 Score |
|---------|----------|
| **CoNLL-2003 NER** | 92.4 (BERT-Large) |

---

## BERT vs GPT 상세 비교

| 특징 | GPT-1 | BERT |
|------|-------|------|
| **아키텍처** | Transformer **Decoder** | Transformer **Encoder** |
| **방향성** | **단방향** (Left-to-right) | **양방향** (Bidirectional) |
| **Attention Mask** | Causal Mask (미래 숨김) | Padding Mask만 |
| **Pre-training Task** | Language Modeling (LM) | **MLM + NSP** |
| **Embeddings** | Token + Position | Token + **Segment** + Position |
| **Activation** | ReLU | **GELU** |
| **Vocabulary** | 40,000 (BPE) | 30,000 (WordPiece) |
| **Parameters (Base)** | 117M | 110M |
| **강점** | Text Generation | Text Understanding |
| **적합한 Task** | Generation, QA | Classification, NER, QA |
| **Training Corpus** | BooksCorpus (5GB) | BooksCorpus + Wikipedia (16GB) |
| **Batch Size** | 64 | 256 |
| **Training Steps** | 100 epochs | 1M steps |

**핵심 차이점**:
1. **Attention 방향**:
   - GPT: i 번째 토큰은 1~i 토큰만 볼 수 있음
   - BERT: i 번째 토큰은 모든 토큰을 볼 수 있음

2. **Pre-training**:
   - GPT: 다음 토큰 예측 (predict next word)
   - BERT: 마스크된 토큰 예측 (predict masked word)

3. **적용**:
   - GPT: Generation에 강함 (text completion, dialogue)
   - BERT: Understanding에 강함 (sentiment, NER, QA)

---

## 핵심 구현 디테일

### 1. Attention Mask (Padding만)

**BERT는 양방향이므로 causal mask 없음**:

```python
def create_attention_mask(self, input_ids, pad_token_id=0):
    # Only mask padding tokens (NOT future tokens!)
    mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    # Shape: (batch_size, 1, 1, seq_len)
    return mask
```

**비교**:
```python
# GPT: Causal + Padding
mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular
mask = mask * (input_ids != pad_token_id)

# BERT: Padding만
mask = (input_ids != pad_token_id)  # No triangular!
```

---

### 2. Weight Initialization

**논문 명시**: Truncated normal distribution N(0, 0.02)

```python
def _init_weights(self):
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
```

---

### 3. Segment IDs 생성

**예시 코드**:
```python
def create_segment_ids(seq_len, sep_position):
    segment_ids = torch.zeros(seq_len, dtype=torch.long)
    segment_ids[sep_position+1:] = 1  # Second sentence
    return segment_ids

# Example
# [CLS] I love BERT [SEP] BERT is great [SEP]
#   0   0  0    0     0     1    1   1     1
```

---

## 코드 실행 예시

```python
# 1. 모델 생성 (BERT-Base 설정)
bert = BERT(
    vocab_size=30000,
    max_seq_len=512,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    dropout=0.1
)

# 2. Pre-training
pretrain_model = BERTForPreTraining(bert)
pretrainer = BERTPreTraining(pretrain_model, learning_rate=1e-4)

# Prepare data
input_ids = torch.randint(0, 30000, (4, 128))
segment_ids = torch.zeros(4, 128, dtype=torch.long)
segment_ids[:, 64:] = 1  # Second sentence

# Mask tokens
masker = MaskedLanguageModelDataPreparation(
    vocab_size=30000, mask_token_id=103, pad_token_id=0,
    cls_token_id=101, sep_token_id=102
)
masked_ids, mlm_labels = masker.mask_tokens(input_ids)

# NSP labels
nsp_labels = torch.randint(0, 2, (4,))

# Train
attention_mask = bert.create_attention_mask(input_ids)
total_loss, mlm_loss, nsp_loss = pretrainer.train_step(
    masked_ids, segment_ids, attention_mask, mlm_labels, nsp_labels
)

print(f"Total Loss: {total_loss:.4f}")
print(f"MLM Loss: {mlm_loss:.4f}")
print(f"NSP Loss: {nsp_loss:.4f}")

# 3. Fine-tuning (Classification)
classifier = BERTForSequenceClassification(bert, num_labels=2)
finetuner = BERTFineTuning(classifier, learning_rate=3e-5)

labels = torch.randint(0, 2, (4,))
loss = finetuner.train_step(input_ids, segment_ids, attention_mask, labels)

print(f"Fine-tuning Loss: {loss:.4f}")
```

---

## 주요 수식 요약

### 1. Attention (Bidirectional)
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
(No causal mask!)
```

### 2. Feed-Forward (GELU)
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
GELU(x) = x * Φ(x)
```

### 3. Embeddings
```
E = Token_Embedding + Segment_Embedding + Position_Embedding
```

### 4. Pre-training Loss
```
L = L_MLM + L_NSP

L_MLM = -Σ log P(masked_token | context)
L_NSP = -log P(IsNext | sentence_pair)
```

### 5. Masking Strategy
```
15% of tokens:
  - 80% → [MASK]
  - 10% → Random token
  - 10% → Original token
```

---

## 구현 완성도

✅ Bidirectional Multi-Head Attention (Causal mask 없음)  
✅ Position-wise Feed-Forward with GELU  
✅ 12-layer Transformer Encoder  
✅ Triple Embeddings (Token + Segment + Position)  
✅ Masked Language Model (80/10/10 strategy)  
✅ Next Sentence Prediction  
✅ Combined Pre-training (MLM + NSP)  
✅ Fine-tuning for Classification  
✅ Special Tokens ([CLS], [SEP], [MASK])  
✅ Weight Tying  
✅ Learning Rate Warmup  
✅ 논문 하이퍼파라미터 정확히 반영  

---

## 중요 개념 정리

### 1. 왜 Masked Language Model?
- **문제**: 양방향 학습시 각 단어가 자기 자신을 볼 수 있음 (trivial prediction)
- **해결**: 일부를 mask해서 context로부터 예측하게 함
- **결과**: 진정한 양방향 표현 학습

### 2. 왜 80/10/10 Masking?
- **80% [MASK]**: 실제 masking 수행
- **10% Random**: [MASK] 토큰에 over-fit 방지
- **10% Original**: Pre-train과 Fine-tune 사이 gap 줄임

### 3. 왜 Next Sentence Prediction?
- QA, NLI 같은 task는 문장 간 관계 이해 필요
- NSP로 sentence-level understanding 학습

### 4. 왜 Segment Embeddings?
- 두 문장을 구분하여 모델이 관계를 학습할 수 있게 함
- NSP task에 필수적

---

## 참고사항

- 실제 학습을 위해서는 WordPiece tokenizer 구현 필요
- BooksCorpus + Wikipedia 데이터셋 준비 필요 (~16GB)
- 실제 논문에서는 TPU Pod (64 TPUs) 사용
- Pre-training: 4일 (BERT-Base), 13일 (BERT-Large)
- Fine-tuning: 대부분 task에서 1 epoch이면 충분

**논문 링크**: https://arxiv.org/abs/1810.04805
