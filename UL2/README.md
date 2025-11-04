# UL2: "Unifying Language Learning Paradigms" 구현

이 프로젝트는 Google의 **"UL2: Unifying Language Learning Paradigms"** 논문에서 제안된 통합 언어 학습 프레임워크를 수식과 개념에 충실하게 구현합니다.

## 📋 논문 개요

**논문**: "UL2: Unifying Language Learning Paradigms"  
**저자**: Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Denny Zhou, Neil Houlsby, Donald Metzler  
**핵심 기여**: 서로 다른 언어 학습 패러다임들을 하나의 통합된 프레임워크로 결합

## 🎯 핵심 아이디어: Mixture of Denoisers (MoD)

### 전체 구조
```
다양한 텍스트 입력
         ↓
   Mode Token 추가 (<R>, <S>, <X>)
         ↓
  해당 모드별 Corruption 적용
         ↓
   Decoder-only Transformer
         ↓
    Target Text 생성
```

## 🔬 세 가지 Denoising 패러다임

### 1. R-Denoiser (Regular Span Corruption)

**BERT-style 학습 패러다임**

**논문 설정**:
- Corruption rate: 15%
- 평균 span length: 3 토큰
- 연속된 span들을 sentinel 토큰으로 대체

**코드 구현**:
```python
def r_denoiser_corruption(text, corruption_rate=0.15, mean_span_length=3.0):
    # Poisson distribution으로 span 길이 샘플링
    span_length = max(1, int(random.expovariate(1.0 / mean_span_length)))
    
    # Sentinel 토큰으로 대체
    input_tokens.append(f"<extra_id_{sentinel_id}>")
    target_tokens.append(f"<extra_id_{sentinel_id}>")
    target_tokens.extend(original_span)
```

**예시**:
```
원본: "The quick brown fox jumps over the lazy dog"
입력: "<R> The quick <extra_id_0> jumps over <extra_id_1> dog"
타겟: "<extra_id_0> brown fox <extra_id_1> the lazy </s>"
```

### 2. S-Denoiser (Sequential Denoising)

**GPT-style 학습 패러다임**

**논문 설정**:
- Prefix 길이: 전체의 50-90%
- Auto-regressive 방식으로 나머지 생성
- Language modeling objective

**코드 구현**:
```python
def s_denoiser_corruption(text, prefix_length=None):
    if prefix_length is None:
        prefix_length = random.randint(
            int(len(text) * 0.5), 
            int(len(text) * 0.9)
        )
    
    input_tokens = text[:prefix_length]
    target_tokens = text[prefix_length:] + ["</s>"]
```

**예시**:
```
원본: "The quick brown fox jumps over the lazy dog"
입력: "<S> The quick brown fox"
타겟: "jumps over the lazy dog </s>"
```

### 3. X-Denoiser (Extreme Denoising)

**극단적인 denoising 작업**

**논문 설정**:
- Corruption rate: 50% (매우 높음)
- 평균 span length: 32 토큰 (매우 길음)
- 더 challenging한 reconstruction 작업

**코드 구현**:
```python
def x_denoiser_corruption(text, corruption_rate=0.5, mean_span_length=32.0):
    # R-denoiser와 같은 방식이지만 더 aggressive한 파라미터
    return r_denoiser_corruption(text, corruption_rate, mean_span_length)
```

**예시**:
```
원본: "The quick brown fox jumps over the lazy dog and runs fast"
입력: "<X> The <extra_id_0> and runs fast"
타겟: "<extra_id_0> quick brown fox jumps over the lazy dog </s>"
```

## 🏗️ 아키텍처 구조

### Decoder-only Transformer (PaLM 스타일)

```python
class TransformerBlock(nn.Module):
    def forward(self, x, attention_mask=None):
        # Pre-normalization structure
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, attention_mask)
        x = x + self.dropout(attn_output)  # Residual connection
        
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)   # Residual connection
        
        return x
```

### RMSNorm (Root Mean Square Normalization)

**논문 수식**:
```
RMSNorm(x) = x / RMS(x) * γ
where RMS(x) = √(mean(x²) + ε)
```

**코드 구현**:
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms * self.weight
        return normalized
```

**LayerNorm과의 차이**:
- LayerNorm: `(x - μ) / σ * γ + β` (평균과 분산 사용)
- RMSNorm: `x / RMS(x) * γ` (평균 제거, bias 없음)

### SwiGLU Activation

**논문 수식**:
```
SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)
where Swish(x) = x * σ(x)
```

**코드 구현**:
```python
class FeedForward(nn.Module):
    def forward(self, x):
        gate = F.silu(self.W_gate(x))  # SiLU = Swish
        up = self.W_up(x)
        hidden = gate * up  # Element-wise multiplication
        output = self.W_down(hidden)
        return output
```

## 🎲 Mode Switching과 Special Tokens

### Mode Tokens
```python
class SpecialTokens:
    R_MODE = "<R>"      # R-Denoiser mode
    S_MODE = "<S>"      # S-Denoiser mode  
    X_MODE = "<X>"      # X-Denoiser mode
```

### Sentinel Tokens
```python
SENTINEL_0 = "<extra_id_0>"
SENTINEL_1 = "<extra_id_1>"
# ... 최대 100개까지
```

**사용 방식**:
1. **Mode token**을 입력 시퀀스 맨 앞에 추가
2. **Sentinel token**으로 corruption된 span 표시
3. 모델이 mode에 따라 다른 학습 목표 수행

## 📊 훈련 비율 (논문에서 제안)

```python
denoiser_ratios = {
    DenoisingMode.R_DENOISER: 0.25,  # 25%
    DenoisingMode.S_DENOISER: 0.25,  # 25% 
    DenoisingMode.X_DENOISER: 0.50   # 50%
}
```

**X-Denoiser가 50%인 이유**:
- 가장 challenging한 작업
- 모델의 robust한 이해 능력 향상
- 다양한 downstream task에 더 잘 일반화

## 🔄 훈련 과정

### 1. 데이터 준비
```python
def prepare_training_data(self, texts):
    for text in texts:
        # 1. 랜덤하게 denoising 모드 선택
        mode = self.sample_denoising_mode()
        
        # 2. 해당 모드에 따라 corruption 적용
        if mode == DenoisingMode.R_DENOISER:
            input_tokens, target_tokens = self.r_denoiser_corruption(tokens)
            mode_token = "<R>"
        # ... S, X 모드도 동일
        
        # 3. Mode token을 맨 앞에 추가
        input_tokens = [mode_token] + input_tokens
```

### 2. Loss 계산
```python
def forward(self, input_ids, labels=None):
    # Decoder-only이므로 next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
```

## 🌟 UL2의 혁신성

### 1. **패러다임 통합**
- **BERT**: Bidirectional context (R-Denoiser로 근사)
- **GPT**: Auto-regressive generation (S-Denoiser)
- **T5**: Span corruption (R, X-Denoiser)

### 2. **Mode-aware Training**
- 하나의 모델이 여러 학습 목표 동시 수행
- Inference시 mode token으로 원하는 동작 지정
- Multi-task learning의 효과

### 3. **Scalability**
- Decoder-only 구조로 scaling 용이
- PaLM 스타일의 최적화된 아키텍처
- 대규모 모델에서 검증됨

## 🎯 Downstream Tasks 적용

### Text Generation
```python
# S-mode로 text generation
input_text = "<S> Once upon a time"
# 모델이 auto-regressive하게 이어서 생성
```

### Text Infilling
```python
# R-mode로 text infilling  
input_text = "<R> The weather is <extra_id_0> today"
# 모델이 빈 자리를 채워서 생성
```

### Summarization
```python
# X-mode로 극단적인 compression
input_text = "<X> [긴 문서] <extra_id_0>"
# 모델이 압축된 요약을 생성
```

## 📈 성능 향상 요인

### 1. **Diverse Training Objectives**
```
R-Denoiser: 양방향 컨텍스트 이해
S-Denoiser: 순차적 생성 능력
X-Denoiser: 극단적 추상화 능력
```

### 2. **Architectural Improvements**
```
RMSNorm: 더 안정적인 학습
SwiGLU: 더 좋은 표현 능력
Pre-norm: Gradient flow 개선
```

### 3. **Unified Framework**
```
하나의 모델로 다양한 태스크 수행
→ 파라미터 효율성 향상
→ Transfer learning 효과 극대화
```

## 🔗 다른 모델과의 비교

| 모델 | 아키텍처 | 학습 목표 | 특징 |
|------|----------|-----------|------|
| **BERT** | Encoder-only | MLM, NSP | 양방향 이해 |
| **GPT** | Decoder-only | Auto-regressive LM | 생성 능력 |
| **T5** | Encoder-Decoder | Span corruption | Text-to-text |
| **UL2** | Decoder-only | MoD (R+S+X) | **통합 패러다임** |

## 🚀 모델 사용법

```python
# 모델 생성
model = create_ul2_model("base")

# 훈련 데이터 준비
trainer = UL2Trainer(model, tokenizer)
training_data = trainer.prepare_training_data(texts)

# 훈련
for input_ids, labels, mode in training_data:
    loss = trainer.train_step(input_ids, labels)

# 추론 (각 모드별)
# R-mode: "<R> The weather is <extra_id_0> today"
# S-mode: "<S> Once upon a time"  
# X-mode: "<X> [document] <extra_id_0>"
```

## 📚 논문의 실험 결과

### SuperGLUE Benchmark
- **UL2-20B**: 89.7점 (당시 SOTA)
- 기존 T5-11B 대비 상당한 성능 향상

### 일반화 능력
- Few-shot learning에서 뛰어난 성능
- 다양한 NLP 태스크에서 일관된 향상
- 특히 reasoning 태스크에서 큰 개선

이 구현은 UL2 논문의 핵심 아이디어인 **Mixture of Denoisers**를 충실히 재현하여, 다양한 언어 학습 패러다임이 어떻게 하나의 통합된 프레임워크로 결합될 수 있는지 보여줍니다.