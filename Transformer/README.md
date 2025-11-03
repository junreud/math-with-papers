# Transformer: "Attention Is All You Need" 구현

이 프로젝트는 Vaswani et al. (2017)의 논문 **"Attention Is All You Need"**에서 제안된 Transformer 아키텍처를 수식과 개념에 충실하게 구현합니다.

## 📋 논문 개요

**논문**: "Attention Is All You Need" (NIPS 2017)  
**저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**핵심 기여**: RNN과 CNN 없이 순전히 attention 메커니즘만으로 구성된 Transformer 아키텍처 제안

## 🏗️ 아키텍처 구조

### 전체 구조: Encoder-Decoder
```
Input Embeddings + Positional Encoding
         ↓
    Encoder Stack (6 layers)
         ↓
    Decoder Stack (6 layers)
         ↓
    Linear + Softmax
         ↓
    Output Probabilities
```

## 🔬 핵심 수식과 구현

### 1. Scaled Dot-Product Attention

**논문 수식**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**코드 구현**:
```python
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    # QK^T / √d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # softmax(QK^T / √d_k)
    attention_weights = F.softmax(scores, dim=-1)
    
    # softmax(...)V
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

**핵심 아이디어**:
- **Q (Query)**: "무엇을 찾고 있는가"
- **K (Key)**: "무엇과 매칭할 것인가"
- **V (Value)**: "실제 정보"
- **√d_k로 스케일링**: gradient vanishing 방지

### 2. Multi-Head Attention

**논문 수식**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**코드 구현**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.d_k = d_model // num_heads  # 논문의 d_k = d_model / h
        
        # 논문의 W^Q_i, W^K_i, W^V_i for all heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 논문의 W^O
        self.W_o = nn.Linear(d_model, d_model)
```

**핵심 아이디어**:
- **병렬 attention**: 서로 다른 representation subspace에서 동시에 attention 수행
- **h=8 heads**: 논문에서 8개의 head 사용
- **d_k = d_v = 64**: d_model=512를 8개 head로 나눔

### 3. Positional Encoding

**논문 수식**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**코드 구현**:
```python
def __init__(self, d_model, max_seq_len=5000):
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    
    # 논문 수식: 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        (-math.log(10000.0) / d_model))
    
    # 논문 수식 적용
    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
```

**핵심 아이디어**:
- **순서 정보 제공**: Self-attention은 순서를 모르므로 위치 정보 필요
- **Sin/Cos 함수**: 상대적 위치 관계를 학습할 수 있음
- **고정 패턴**: 학습이 아닌 수학적 함수로 생성

### 4. Position-wise Feed-Forward Networks

**논문 수식**:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**코드 구현**:
```python
class PositionwiseFeedForward(nn.Module):
    def forward(self, x):
        # max(0, xW_1 + b_1) - ReLU activation
        hidden = F.relu(self.W_1(x))
        
        # max(0, xW_1 + b_1)W_2 + b_2
        output = self.W_2(hidden)
        return output
```

**핵심 아이디어**:
- **각 위치별로 동일한 연산**: 모든 position에 같은 FFN 적용
- **두 번의 선형 변환**: 512 → 2048 → 512
- **ReLU 활성화**: 비선형성 제공

## 🔄 Layer 구조와 Residual Connection

### Encoder Layer
```
x → Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm → output
```

### Decoder Layer
```
x → Masked Multi-Head Self-Attention → Add & Norm
  → Multi-Head Cross-Attention → Add & Norm  
  → FFN → Add & Norm → output
```

**코드 구현**:
```python
# Encoder Layer
attn_output = self.self_attn(x, x, x, src_mask)
x = self.norm1(x + self.dropout(attn_output))  # Add & Norm

ff_output = self.feed_forward(x)
x = self.norm2(x + self.dropout(ff_output))    # Add & Norm

# Decoder Layer (추가로 Cross-Attention)
cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
x = self.norm2(x + self.dropout(cross_attn_output))  # Add & Norm
```

## 🎯 Attention의 종류

### 1. Encoder Self-Attention
- **입력**: 동일한 소스 시퀀스
- **특징**: 양방향으로 모든 위치를 볼 수 있음
- **용도**: 입력 문장의 내부 관계 파악

### 2. Decoder Masked Self-Attention
- **입력**: 타겟 시퀀스 (causal mask 적용)
- **특징**: 현재 위치 이전만 볼 수 있음 (auto-regressive)
- **용도**: 생성 중인 문장의 이전 컨텍스트 활용

### 3. Encoder-Decoder Attention (Cross-Attention)
- **Query**: Decoder의 출력
- **Key, Value**: Encoder의 출력
- **용도**: 소스 문장과 타겟 문장 간의 관계 파악

## 📊 하이퍼파라미터 (논문 기본 설정)

| 파라미터 | 값 | 설명 |
|---------|---|------|
| d_model | 512 | 모델 차원 |
| N (layers) | 6 | Encoder/Decoder 레이어 수 |
| h (heads) | 8 | Multi-head attention의 head 수 |
| d_k, d_v | 64 | Key, Value 차원 (d_model/h) |
| d_ff | 2048 | Feed-forward 내부 차원 |
| dropout | 0.1 | 드롭아웃 비율 |

## 🚀 모델 사용법

```python
# 모델 생성 (논문 기본 설정)
model = create_transformer_model(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048
)

# 입력 준비
src = torch.randint(1, 1000, (batch_size, src_seq_len))  # 소스 문장
tgt = torch.randint(1, 1000, (batch_size, tgt_seq_len))  # 타겟 문장

# Forward pass
output = model(src, tgt)  # (batch_size, tgt_seq_len, vocab_size)
```

## 🎲 Mask의 종류와 역할

### 1. Padding Mask
```python
def create_padding_mask(seq, pad_idx=0):
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(1)
    return mask
```
- **목적**: 패딩 토큰에 attention 주지 않기
- **적용**: Encoder와 Decoder 모두

### 2. Causal Mask (Look-ahead Mask)
```python
def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return (mask == 0).unsqueeze(0).unsqueeze(0)
```
- **목적**: 미래 토큰 참조 방지
- **적용**: Decoder의 self-attention만

## 🔄 Training과 Inference

### Training
- **Teacher Forcing**: 실제 타겟 시퀀스를 입력으로 사용
- **Loss**: Cross-entropy loss
- **Optimizer**: Adam with learning rate scheduling

### Inference
- **Auto-regressive**: 한 번에 하나씩 토큰 생성
- **Beam Search**: 여러 후보 중 최적 선택

## 🌟 논문의 혁신성

1. **RNN/CNN 제거**: 순전히 attention만으로 구성
2. **병렬화 가능**: RNN처럼 순차적이지 않아 훈련 속도 향상
3. **Long-range Dependencies**: 거리에 관계없이 직접적인 연결
4. **범용성**: 다양한 sequence-to-sequence 태스크에 적용 가능

## 🔗 후속 연구에 미친 영향

- **BERT**: Encoder-only Transformer
- **GPT**: Decoder-only Transformer  
- **T5**: Text-to-Text Transfer Transformer
- **Vision Transformer**: 이미지 분야 적용
- **Switch Transformer**: Sparse MoE 적용

이 구현은 논문의 수식과 아키텍처를 최대한 충실하게 재현하여, Transformer의 핵심 개념을 이해할 수 있도록 도와줍니다.