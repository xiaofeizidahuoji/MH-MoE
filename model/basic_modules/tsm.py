import torch
import torch.nn as nn


class TinyTimeMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, drop, hidden_dim=None):
        super().__init__()
        num_hidden = out_features if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, inputs: torch.Tensor):
        """
            inputs (`torch.Tensor` of shape `((batch_size, num_patches, num_channels, d_model))`):
                Input to the MLP layer.
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TinyTimeMixerGatedAttention(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs



class FeatureMixerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop, use_gated, output_dim=None):
        super().__init__()
        if output_dim == None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim, eps=1e-5)
        self.gated_attn = use_gated
        self.mlp = TinyTimeMixerMLP(
            in_features=input_dim,
            out_features=self.output_dim,
            hidden_dim=hidden_dim,
            drop=drop
        )
        if use_gated:
            self.gating_block = TinyTimeMixerGatedAttention(in_size=self.output_dim, out_size=self.output_dim)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, n_channel, d_model)`):
                Input tensor to the layer.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)
        if self.gated_attn:
            hidden = self.gating_block(hidden)
        if self.output_dim == 1:  # reg -> pred
            return hidden
        out = hidden + residual
        return out
