# T5: Text-to-Text Transfer Transformer

**논문**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2019)

T5는 모든 NLP 태스크를 "텍스트 입력 → 텍스트 출력" 형식으로 통일한 혁신적인 접근법을 제시했습니다.

## 🎯 핵심 아이디어

### 1. Text-to-Text Framework
모든 언어 태스크를 동일한 형식으로 처리:

```
# Translation
Input:  "translate English to German: Hello world"
Output: "Hallo Welt"

# Summarization  
Input:  "summarize: [long article text]"
Output: "[summary]"

# Question Answering
Input:  "question: What is the capital? context: Paris is the capital of France."
Output: "Paris"

# Classification
Input:  "cola sentence: This sentence is grammatical."
Output: "acceptable"
```

### 2. 통일된 아키텍처
- **하나의 모델**로 모든 NLP 태스크 처리
- **동일한 loss function** (cross-entropy)
- **동일한 decoding 방식**
- **Multi-task learning** 자연스럽게 가능

## 🏗️ 모델 아키텍처

### 논문의 수식과 구현 매핑

#### 1. Relative Position Encoding
**논문 수식**: Attention에 relative position bias 추가
```
A_ij = Q_i · K_j + b_{clip(i-j, -k, k)}
```

**구현**:
```python
class RelativePositionBias(nn.Module):
    def forward(self, query_length: int, key_length: int):
        # 상대적 위치 계산: i - j
        relative_position = memory_position - context_position
        
        # Bucket으로 변환 (가까운 거리는 세밀하게, 먼 거리는 coarse하게)
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        # Bias 계산
        bias = self.relative_attention_bias(relative_position_bucket)
        return bias
```

**핵심 인사이트**:
- 절대 위치 대신 **상대적 위치 관계** 학습
- **로그 스케일 bucketing**으로 효율적 처리
- **첫 번째 layer에만** relative bias 적용

#### 2. T5 Layer Normalization
**논문**: RMSNorm과 유사하지만 centering 없음
```
LayerNorm(x) = x / sqrt(variance + ε) * scale
```

**구현**:
```python
class T5LayerNorm(nn.Module):
    def forward(self, hidden_states):
        # T5는 mean을 빼지 않고 variance로만 normalize
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
```

**차이점**:
- BERT/GPT: `(x - mean) / sqrt(variance + ε)`
- T5: `x / sqrt(variance + ε)` (mean centering 없음)

#### 3. Multi-Head Attention with Relative Bias
**논문 수식**:
```
Attention(Q, K, V) = softmax((QK^T + bias) / sqrt(d_k))V
```

**구현**:
```python
def forward(self, query, key, value, position_bias=None):
    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
    # Add relative position bias
    if position_bias is not None:
        scores += position_bias
    
    # Softmax and apply to values
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
```

#### 4. Pre-Normalization Structure
**T5 특징**: Layer norm이 attention/FFN **이전**에 적용

```python
def forward(self, hidden_states):
    # Pre-norm for self-attention
    norm_hidden_states = self.layer_norm_1(hidden_states)
    attention_output = self.self_attention(norm_hidden_states)
    hidden_states = hidden_states + attention_output  # residual
    
    # Pre-norm for feed-forward  
    norm_hidden_states = self.layer_norm_2(hidden_states)
    ff_output = self.feed_forward(norm_hidden_states)
    hidden_states = hidden_states + ff_output  # residual
```

## 🎲 Pre-training: Span Corruption

### 핵심 아이디어
BERT의 masked language modeling을 **연속된 span**으로 확장

### 알고리즘
1. **15% 토큰**을 corruption 대상으로 선택
2. **평균 3 토큰**의 연속된 span으로 그룹화
3. 각 span을 **sentinel 토큰**으로 대체
4. 모델이 sentinel 순서대로 **원본 토큰들을 예측**

### 예시
```python
# 원본 텍스트
text = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

# Span corruption 적용 (noise_density=0.33, mean_span_length=3)
corrupted = ["The", "<extra_id_0>", "fox", "<extra_id_1>", "the", "lazy", "dog"]
target = ["<extra_id_0>", "quick", "brown", "<extra_id_1>", "jumps", "over", "<extra_id_2>"]

# 학습 형태
input_text = "The <extra_id_0> fox <extra_id_1> the lazy dog"
target_text = "<extra_id_0> quick brown <extra_id_1> jumps over <extra_id_2>"
```

### 구현
```python
class SpanCorruption:
    @staticmethod
    def corrupt_spans(text, noise_density=0.15, mean_noise_span_length=3.0):
        # 마스킹할 토큰 수 계산
        num_noise_tokens = int(round(len(text) * noise_density))
        
        # 평균 span 길이를 기반으로 span 수 계산  
        num_noise_spans = max(1, round(num_noise_tokens / mean_noise_span_length))
        
        # 지수분포에서 각 span 길이 샘플링
        # 랜덤 위치에 span 배치
        # Sentinel 토큰으로 대체
        
        return corrupted_tokens, target_tokens
```

## 📊 모델 크기별 Configuration

| Model | Parameters | d_model | Layers | Heads | d_ff |
|-------|-----------|---------|--------|-------|------|
| T5-Small | 60M | 512 | 6 | 8 | 2,048 |
| T5-Base | 220M | 768 | 12 | 12 | 3,072 |
| T5-Large | 770M | 1024 | 24 | 16 | 4,096 |
| T5-3B | 3B | 1024 | 24 | 32 | 16,384 |
| T5-11B | 11B | 1024 | 24 | 128 | 65,536 |

## 🚀 사용 방법

### 1. 모델 생성
```python
from main import create_t5_model

# 다양한 크기의 모델 생성 가능
model = create_t5_model("base")  # 220M parameters
```

### 2. Text-to-Text 형식으로 학습
```python
# 번역 태스크
input_text = "translate English to German: Hello world"
target_text = "Hallo Welt"

# 요약 태스크  
input_text = "summarize: [긴 문서 내용]"
target_text = "[요약된 내용]"

# Forward pass
logits, loss = model(input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
```

### 3. 생성 (Generation)
```python
# Greedy decoding으로 텍스트 생성
generated_ids = model.generate(
    input_ids=input_ids,
    max_length=50,
    do_sample=False  # greedy
)
```

## 🔍 T5의 혁신적 기여

### 1. Unified Framework
- **모든 NLP 태스크**를 동일한 형식으로 처리
- **태스크별 헤드 불필요** (모두 text generation)
- **Multi-task learning** 자연스럽게 지원

### 2. Transfer Learning의 체계적 분석
논문에서 체계적으로 연구한 요소들:
- **Pre-training objectives** (MLM vs span corruption vs autoregressive)
- **Architectures** (encoder-decoder vs decoder-only)
- **Unlabeled datasets** (C4, Web crawl 등)
- **Transfer approaches** (fine-tuning vs multi-task)
- **Model sizes** (Small부터 11B까지)

### 3. C4 Dataset
**Colossal Clean Crawled Corpus**:
- Common Crawl 기반
- 750GB의 필터링된 영어 텍스트
- Deduplication과 quality filtering 적용

### 4. Relative Position Encoding
- **절대 위치의 한계** 극복
- **상대적 거리**에 기반한 attention bias
- **긴 시퀀스**에서도 효과적

## 📈 성능 및 영향

### GLUE/SuperGLUE 성능
T5-11B는 당시 **SOTA** 달성:
- GLUE: 90.3점
- SuperGLUE: 89.3점

### 후속 모델들에 미친 영향
1. **PaLM, PaLM-2**: Text-to-text paradigm 계승
2. **UL2**: T5의 span corruption을 더욱 발전
3. **mT5**: Multilingual 확장
4. **ByT5**: Byte-level tokenization

## 🎯 핵심 교훈

### 1. "Everything is Text-to-Text"
NLP의 모든 문제를 **text generation**으로 통일할 수 있다는 통찰

### 2. Scale + Transfer Learning
**큰 모델** + **좋은 pre-training** + **효과적인 transfer**의 조합

### 3. Systematic Evaluation
단순히 좋은 결과가 아닌, **각 구성요소의 기여도**를 체계적으로 분석

### 4. Simplicity is Power
복잡한 task-specific 구조 대신 **단순하고 통일된 접근법**의 효과

## 💡 구현의 핵심 포인트

1. **Relative Position Bias**: 첫 번째 layer에만 적용, 나머지는 재사용
2. **T5LayerNorm**: Mean centering 없이 variance만으로 정규화
3. **Pre-normalization**: Layer norm이 attention/FFN 이전에 적용
4. **Shared Embeddings**: Input과 output embedding 공유
5. **Span Corruption**: 연속된 토큰들을 sentinel로 대체하는 pre-training

T5는 **Transfer Learning의 새로운 패러다임**을 제시하며, 현재까지도 많은 모델들의 기반이 되고 있는 중요한 연구입니다.