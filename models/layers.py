import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, projection_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1,
                                                                                2)  # [batch_size,num_classes, seq_len]
        weighted_output = att_weights @ x  # [batch_size,num_classes, input_size]
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and reshape query, key, and value
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Broadcast mask to (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Fill masked positions with -inf

        # Compute attention probabilities
        attn_probs = nn.functional.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attended values
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Project attended values
        output = self.out_proj(attn_output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=384, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src, mask=None):
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask=mask)
        return src


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        attn_output = self.multihead_attn(query, key, value, mask=mask)
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output


class TransformerCrossAttn(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerCrossAttn, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.cross_attn_layers = nn.ModuleList([CrossAttention(d_model, nhead, dropout) for _ in range(num_layers)])
        self.feed_forward_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        ) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x1, x2, mask=None):
        x1_pos = self.pos_encoder(x1)
        x2_pos = self.pos_encoder(x2)

        for i in range(len(self.cross_attn_layers)):
            # x1 attend to x2
            x1_pos = self.cross_attn_layers[i](query=x2_pos, key=x1_pos, value=x1_pos, mask=mask)
            x1_pos = x1_pos + self.feed_forward_layers[i](x1_pos)
            x1_pos = self.norm_layers[i](x1_pos)

            # x2 attend to x1
            x2_pos = self.cross_attn_layers[i](query=x1_pos, key=x2_pos, value=x2_pos, mask=mask)
            x2_pos = x2_pos + self.feed_forward_layers[i](x2_pos)
            x2_pos = self.norm_layers[i](x2_pos)

        return x1_pos, x2_pos


class ContrastiveLearningLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLearningLoss, self).__init__()
        self.temperature = temperature

    def _contrastive_loss(self, z1, z2):
        batch_size = z1.size(0)

        global_repr1 = z1.mean(dim=1)  # (batch_size, hidden_dim)
        global_repr2 = z2.mean(dim=1)  # (batch_size, hidden_dim)

        pos_sim = torch.einsum('bd,bd->b', global_repr1, global_repr2)  # (batch_size,)
        neg_sim = torch.einsum('bd,cd->bc', global_repr1, global_repr2)  # (batch_size, batch_size)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, batch_size+1)
        logits /= self.temperature

        labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, z1, z2):
        loss_z1_z2 = self._contrastive_loss(z1, z2)
        loss_z2_z1 = self._contrastive_loss(z2, z1)
        loss = (loss_z1_z2 + loss_z2_z1) / 2
        return loss


class MultiViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=10):
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature

    def compute_joint(self, x_out, x_tf_out):
        # produces variable that requires grad (since args require grad)

        bn, k = x_out.size()
        assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def forward(self, x_out, x_tf_out, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        if len(x_out.size()) == 3:
            x_out = x_out.mean(dim=1)  # (batch_size, hidden_dim)
            x_tf_out = x_tf_out.mean(dim=1)  # (batch_size, hidden_dim)

        x_out, x_tf_out = F.softmax(x_out, dim=-1), F.softmax(x_tf_out, dim=-1)
        _, k = x_out.size()

        p_i_j = self.compute_joint(x_out, x_tf_out)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
        p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

        loss = - p_i_j * (torch.log(p_i_j) \
                          - self.temperature * torch.log(p_j) \
                          - self.temperature * torch.log(p_i))

        loss = loss.sum()

        return loss