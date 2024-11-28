import torch
import torch.nn as nn
import torch.nn.functional as F

class MHMoE(nn.Module):
    def __init__(self, args):
        super(MHMoE, self).__init__()
        
        # Initialize configuration parameters
        self.num_nodes = args.num_nodes
        self.base_input_dim = args.input_base_dim
        self.extra_input_dim = args.input_extra_dim
        self.output_dim = args.output_dim
        self.history_length = args.his
        self.prediction_length = args.pred
        self.embedding_dim = args.embed_dim
        self.mode = args.mode
        self.model_list = args.model_list
        self.predictors = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        self.args = args
        self.topk = args.topk

        # Determine input dimensions based on mode
        input_dim = self.embedding_dim * 4 if self.mode != 'ori' else self.base_input_dim
        args.dim_in = input_dim

        # Initialize predictors
        for model_name in self.model_list:
            if 'STID' in model_name:
                from model.STID.STID import STID
                from model.STID.args import parse_args
                predictor_args = parse_args(args.dataset_use)
                self.predictors[model_name] = STID(predictor_args, args)
            elif 'GWN' in model_name:
                from model.GWN.GWN import GWNET
                from model.GWN.args import parse_args
                predictor_args = parse_args(args.dataset_use)
                predictor_args.device = args.device
                self.predictors[model_name] = GWNET(predictor_args, input_dim, self.output_dim, args.A_dict, [args.dataset_use], self.mode)
            elif 'STWave' in model_name:
                from model.STWave.STWave import STWave
                from model.STWave.args import parse_args
                predictor_args = parse_args(args.dataset_use, args)
                self.predictors[model_name] = STWave(predictor_args, args)

        # Freeze model parameters for certain modes
        if self.mode in ['test', 'train']:
            for model_name in self.model_list:
                for param in self.predictors[model_name].parameters():
                    param.requires_grad = False

                if self.mode == 'train':
                    if 'STID' in model_name:
                        self._set_trainable_params(self.predictors[model_name], ['time_series_emb_layer', 'regression_layer'])
                    elif 'GWN' in model_name:
                        self._set_trainable_params(self.predictors[model_name], ['start_conv', 'end_conv_2'])
                    elif 'STWave' in model_name:
                        self._set_trainable_params(self.predictors[model_name], ['end_emb'])

        # Xavier initialization for trainable parameters
        if self.mode in ['train', 'ori'] and args.xavier:
            for model_name in self.model_list:
                for param in self.predictors[model_name].parameters():
                    if param.dim() > 1 and param.requires_grad:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param)

        if self.mode != 'ori':
            from Extractor import Extractor
            self.extractor = Extractor(args)

            from model.basic_modules import Adapter, MemoryGating_class
            for model_name in self.model_list:
                self.adapters[model_name] = Adapter(args)

            args.gate_dim = len(self.model_list)
            self.memory_gating = MemoryGating_class(args)
            
            if self.mode == 'test':  
                for param in self.extractor.parameters():
                    param.requires_grad = False
                for model in self.model_list:
                    for param in self.adapters[model].parameters():
                        param.requires_grad = False
                for param in self.memory_gating.parameters():
                    param.requires_grad = False


    def _set_trainable_params(self, module, param_names):
        """Helper function to set specific parameters of a module as trainable."""
        for param_name in param_names:
            for param in getattr(module, param_name).parameters():
                param.requires_grad = True

    def forward(self, source, label, dataset_selection, **kwargs):
        if self.mode == 'test':
            return self._forward_train_mem(source, label, dataset_selection, **kwargs)
        elif self.mode == 'train':
            return self._forward_train_mem(source, label, dataset_selection, **kwargs)

    def _forward_train_mem(self, source, label, dataset_selection, batch_seen=None, nadj=None, lpls=None, useGNN=False, DSU=True, epoch=20):
        if torch.cuda.device_count() > 1:
            lpls, nadj = lpls.squeeze(0), nadj.squeeze(0)

        # Extract pattern embeddings
        pattern_embeddings = self.extractor(source[..., :self.base_input_dim], source, None, nadj, lpls, useGNN)

        # Adapter embeddings
        adapter_outputs = []
        for model_name in self.model_list:
            adapted_embedding = self.adapters[model_name](pattern_embeddings)
            adapter_outputs.append(adapted_embedding)

        # Gating mechanism
        adapter_stack = torch.stack(adapter_outputs, dim=0)
        gate_outputs, gating_states = self.memory_gating(pattern_embeddings, adapter_stack, source)
        mem_retrieved, label_retrieved, mem_label, negative_samples, attention_weights = gating_states

        # Random gate initialization during explore stage
        if epoch < self.args.explore_stage:
            gate_outputs = torch.randn_like(gate_outputs)

        # Predictions from all experts
        all_predictions = []
        for i, model_name in enumerate(self.model_list):
            adapted_input = adapter_outputs[i] + source[..., :self.base_input_dim]
            if 'STID' in model_name:
                model_input = torch.cat([adapted_input, source], dim=-1)
                prediction = self.predictors[model_name](model_input, dataset_selection)
            elif 'STWave' in model_name:
                prediction = self.predictors[model_name](source, None, dataset_selection, adapted_input)
            else:
                prediction = self.predictors[model_name](adapted_input, dataset_selection)
            all_predictions.append(prediction[None, :])

        all_predictions = torch.cat(all_predictions, dim=0)  # Combine predictions [n_models, B, T, N, 1]

        # Top-k expert selection
        if epoch < self.args.explore_stage:  # During explore stage
            top_k = 1
            gate_topk, topk_indices = torch.topk(gate_outputs, k=top_k, dim=-1)
            gate_topk = gate_topk.softmax(dim=-1)
        else:
            top_k = self.topk
            gate_topk, topk_indices = torch.topk(gate_outputs, k=top_k, dim=-1)

        # Gather top-k predictions
        topk_indices = topk_indices.unsqueeze(0).unsqueeze(2).unsqueeze(4)
        all_predictions_expanded = all_predictions.unsqueeze(-1).expand(-1, -1, -1, -1, -1, top_k)
        topk_indices_expanded = topk_indices.expand(-1, -1, all_predictions.shape[2], -1, all_predictions.shape[4], -1)
        topk_predictions = torch.gather(all_predictions_expanded, 0, topk_indices_expanded).squeeze(0)

        # Weighted sum of top-k predictions
        gate_weights = gate_topk.unsqueeze(1).unsqueeze(3).expand_as(topk_predictions)
        final_predictions = torch.sum(gate_weights * topk_predictions, dim=-1)

        # Ensure outputs match original
        return final_predictions, all_predictions, label_retrieved

