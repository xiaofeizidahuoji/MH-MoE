import torch
from torch import nn
from basic_modules.tsm import FeatureMixerBlock

class GCN(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=hidden_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, input_data: torch.Tensor, nadj: torch.Tensor, useGNN=False) -> torch.Tensor:
        if useGNN:
            gcn_out = self.act(torch.einsum('nk,bdke->bdne', nadj, self.fc1(input_data)))
        else:
            gcn_out = self.act(self.fc1(input_data))
        return gcn_out + input_data



class Extractor(nn.Module):
    def __init__(self, args):
        super(Extractor, self).__init__()
        self.args = args
        self.mode = args.mode
        self.spatial_dim = args.node_dim
        self.seq_length = args.his
        self.embedding_dim = args.embed_dim
        self.pred_length = args.pred
        self.temporal_tid_dim = args.temp_dim_tid
        self.temporal_diw_dim = args.temp_dim_diw

        # Feature inclusion flags
        self.include_tid = args.if_time_in_day
        self.include_diw = args.if_day_in_week
        self.include_spatial = args.if_spatial

        self.base_input_dim = args.input_base_dim

        # Hidden dimensions calculation
        self.hidden_dim = (
            self.embedding_dim
            + self.spatial_dim * int(self.include_spatial)
            + self.temporal_tid_dim * int(self.include_tid)
            + self.temporal_diw_dim * int(self.include_diw)
        )
        
        # Spatial embedding layers
        if self.include_spatial:
            self.spatial_layer_1 = nn.Linear(self.spatial_dim, self.spatial_dim)
            self.spatial_layer_2 = nn.Linear(self.spatial_dim, self.spatial_dim)

        # Temporal embedding layers
        if self.include_tid:
            self.time_in_day_embedding = nn.Embedding(288 + 1, self.temporal_tid_dim)
        if self.include_diw:
            self.day_in_week_embedding = nn.Embedding(7 + 1, self.temporal_diw_dim)

        # Time-series embedding
        self.timeseries_embed_layer = nn.Linear(self.seq_length, self.embedding_dim, bias=True)

        # Feature mixing encoders
        self.use_gated = True
        self.encoder_stage_1 = nn.Sequential(
            FeatureMixerBlock(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim // 2, drop=0.1, use_gated=self.use_gated),
            FeatureMixerBlock(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim // 2, drop=0.1, use_gated=self.use_gated),
        )
        self.encoder_stage_2 = nn.Sequential(
            FeatureMixerBlock(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim // 2, drop=0.1, use_gated=self.use_gated),
            FeatureMixerBlock(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim // 2, drop=0.1, use_gated=self.use_gated),
        )
        
        # Graph convolution layers
        self.graph_conv_1 = GCN(self.hidden_dim)
        self.graph_conv_2 = GCN(self.hidden_dim)

        # Activation function
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, history_data, source2, batch_seen=None, nadj=None, lpls=None, useGNN=False, DSU=True):
        # Input processing
        batch_size, _, num_nodes, _ = history_data.shape
        zero_tensor = torch.IntTensor(1).to(self.args.device)

        # Temporal embeddings
        tid_indices = source2[:, 0, :, self.base_input_dim]
        tid_embedding = self.time_in_day_embedding(tid_indices.type_as(zero_tensor).long())
        diw_indices = source2[:, 0, :, self.base_input_dim + 1]
        diw_embedding = self.day_in_week_embedding(diw_indices.type_as(zero_tensor).long())

        # Time-series embedding
        timeseries_embedding = self.timeseries_embed_layer(
            history_data[..., 0:self.base_input_dim].transpose(1, 3)
        )

        # Spatial embeddings
        spatial_data = self.spatial_layer_1(lpls)
        spatial_encoded = self.leaky_relu(spatial_data)
        spatial_expanded = self.spatial_layer_2(spatial_encoded).unsqueeze(0).expand(batch_size, -1, -1)
        spatial_expanded = spatial_expanded.unsqueeze(1).repeat(1, self.base_input_dim, 1, 1)
        spatial_features = [spatial_expanded]

        # Combine temporal embeddings
        temporal_features = []
        temporal_features.append(tid_embedding.unsqueeze(1))
        temporal_features.append(diw_embedding.unsqueeze(1))

        # Concatenate features
        combined_features = torch.cat([timeseries_embedding] + spatial_features + temporal_features, dim=-1)
        combined_features = combined_features.transpose(1, 3)

        # Encoding with GCN and FeatureMixer
        gcn_output_1 = self.graph_conv_1(combined_features, nadj, useGNN)
        encoded_stage_1 = self.encoder_stage_1(gcn_output_1.transpose(1, 3)).transpose(1, 3)

        gcn_output_2 = self.graph_conv_2(encoded_stage_1, nadj, useGNN)
        final_encoding = self.encoder_stage_2(gcn_output_2.transpose(1, 3)).transpose(1, 3)

        # Output
        output_tensor = final_encoding.transpose(1, 3)
        return output_tensor

