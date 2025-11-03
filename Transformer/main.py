"""
Transformer Implementation based on "Attention Is All You Need"
논문의 핵심 수식과 개념을 충실히 구현

주요 구성 요소:
1. Encoder-Decoder 아키텍처
2. Multi-Head Self-Attention
3. Position-wise Feed-Forward Networks
4. Positional Encoding
5. Residual Connections and Layer Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 메커니즘
    논문 수식: 
    - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    - MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    - head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 논문의 d_k = d_model / h
        
        # Q, K, V projection matrices (논문의 W^Q_i, W^K_i, W^V_i for all heads)
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
        
        Args:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
        """
        
        # QK^T / sqrt(d_k) - 논문 수식의 핵심
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Mask 적용 (padding mask 또는 causal mask)
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
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: attention mask
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear projections: QW^Q, KW^K, VW^V
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_v, d_model)
        
        # Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concat heads: (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Final linear projection: Concat(head_1, ..., head_h)W^O
        output = self.W_o(attn_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks
    논문 수식: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        논문 수식: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        """
        # max(0, xW_1 + b_1) - ReLU activation
        hidden = F.relu(self.W_1(x))
        hidden = self.dropout(hidden)
        
        # max(0, xW_1 + b_1)W_2 + b_2
        output = self.W_2(hidden)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    논문 수식:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 논문 수식: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 논문 수식 적용
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].transpose(0, 1)  # Broadcasting


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    논문 구조: LayerNorm(x + MultiHeadAttention(x)) -> LayerNorm(x + FFN(x))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization (논문에서는 Add & Norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            src_mask: (batch_size, 1, 1, seq_len)
        """
        # Multi-Head Self-Attention with residual connection and layer norm
        # LayerNorm(x + MultiHeadAttention(x))
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Position-wise Feed-Forward with residual connection and layer norm
        # LayerNorm(x + FFN(x))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    논문 구조: 
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention (Encoder-Decoder Attention)
    3. Position-wise Feed-Forward
    각각 residual connection과 layer normalization
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output: encoder output (batch_size, src_seq_len, d_model)
            tgt_mask: causal mask for decoder self-attention
            src_mask: padding mask for encoder-decoder attention
        """
        # 1. Masked Multi-Head Self-Attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Multi-Head Cross-Attention (Encoder-Decoder Attention)
        # Query from decoder, Key and Value from encoder
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Position-wise Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    N개의 동일한 encoder layer를 쌓은 구조
    """
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            src_mask: (batch_size, 1, 1, seq_len)
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    N개의 동일한 decoder layer를 쌓은 구조
    """
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output: encoder output (batch_size, src_seq_len, d_model)
            tgt_mask: causal mask for decoder
            src_mask: padding mask for encoder
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x


class Transformer(nn.Module):
    """
    Complete Transformer Model
    논문의 전체 아키텍처 구현: Encoder-Decoder with Attention
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters (논문에서 제안한 방법)
        self._init_parameters()
        
    def _init_parameters(self):
        """
        논문에서 제안한 parameter initialization
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Padding mask 생성
        Args:
            seq: (batch_size, seq_len)
            pad_idx: padding token index
        Returns:
            mask: (batch_size, 1, 1, seq_len)
        """
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(1)
        return mask
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """
        Causal mask 생성 (decoder용)
        Args:
            size: sequence length
        Returns:
            mask: (1, 1, size, size)
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return (mask == 0).unsqueeze(0).unsqueeze(0)
    
    def encode(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder forward pass
        Args:
            src: (batch_size, src_seq_len)
            src_mask: (batch_size, 1, 1, src_seq_len)
        """
        # Embedding + Positional Encoding
        # 논문 수식: Embedding(x) * sqrt(d_model) + PositionalEncoding(x)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)  # (batch_size, seq_len, d_model)
        src_emb = self.dropout(src_emb)
        
        # Encoder
        encoder_output = self.encoder(src_emb, src_mask)
        
        return encoder_output
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder forward pass
        Args:
            tgt: (batch_size, tgt_seq_len)
            encoder_output: (batch_size, src_seq_len, d_model)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
            src_mask: (batch_size, 1, 1, src_seq_len)
        """
        # Embedding + Positional Encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # (batch_size, seq_len, d_model)
        tgt_emb = self.dropout(tgt_emb)
        
        # Decoder
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        
        return decoder_output
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0
    ) -> torch.Tensor:
        """
        Complete forward pass
        Args:
            src: (batch_size, src_seq_len)
            tgt: (batch_size, tgt_seq_len)
        Returns:
            output: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks
        src_mask = self.create_padding_mask(src, src_pad_idx)
        tgt_seq_len = tgt.size(1)
        tgt_causal_mask = self.create_causal_mask(tgt_seq_len).to(tgt.device)
        tgt_pad_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output


def create_transformer_model(
    src_vocab_size: int = 10000,
    tgt_vocab_size: int = 10000,
    d_model: int = 512,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    d_ff: int = 2048,
    max_seq_len: int = 5000,
    dropout: float = 0.1
) -> Transformer:
    """
    논문의 기본 설정으로 Transformer 모델 생성
    
    논문 기본 설정:
    - d_model = 512
    - num_heads = 8 (따라서 d_k = d_v = 64)
    - num_layers = 6 (encoder와 decoder 각각)
    - d_ff = 2048
    - dropout = 0.1
    """
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )


# 예제 사용법
if __name__ == "__main__":
    # 논문의 기본 설정으로 모델 생성
    model = create_transformer_model()
    
    print("Transformer Model Architecture:")
    print(f"- Model dimension (d_model): {model.d_model}")
    print(f"- Number of attention heads: 8")
    print(f"- Number of encoder layers: 6")
    print(f"- Number of decoder layers: 6")
    print(f"- Feed-forward dimension (d_ff): 2048")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 예제 입력
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # 더미 데이터
    src = torch.randint(1, 1000, (batch_size, src_seq_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_seq_len))
    
    print(f"\nExample forward pass:")
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Output shape: {output.shape}")
        print(f"Output logits for next token prediction")
        
    print("\n논문의 핵심 구현 요소:")
    print("1. ✅ Scaled Dot-Product Attention: softmax(QK^T/√d_k)V")
    print("2. ✅ Multi-Head Attention: Concat(head_1,...,head_h)W^O")
    print("3. ✅ Positional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d_model))")
    print("4. ✅ Encoder-Decoder Architecture with 6 layers each")
    print("5. ✅ Residual Connections and Layer Normalization")
    print("6. ✅ Position-wise Feed-Forward: FFN(x) = max(0,xW_1+b_1)W_2+b_2")