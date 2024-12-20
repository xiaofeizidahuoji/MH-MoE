import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order*support_len+1)*c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(nn.Module):
    def __init__(self, args, dim_in, dim_out, A_dict, dataset_use, mode):
        super(GWNET, self).__init__()
        self.adj_mx = args.adj_mx
        self.mode = mode
        self.num_nodes = args.num_nodes
        self.feature_dim = dim_in

        self.dropout = args.dropout
        self.blocks = args.blocks
        self.layers =  args.layers
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj
        self.adjtype = args.adjtype
        self.randomadj = args.randomadj
        self.aptonly = args.aptonly
        self.kernel_size = args.kernel_size
        self.nhid = args.nhid
        self.residual_channels = args.residual_channels
        self.dilation_channels = args.dilation_channels
        self.skip_channels = self.nhid * 8
        self.end_channels = self.nhid * 16
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.output_dim = dim_out
        self.device = args.device

        self._logger = getLogger()

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        if self.adj_mx is None:
            self.supports = None
        else:
            self.cal_adj(self.adjtype)
            self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        if self.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0]
        if self.aptonly:
            self.supports = None

        receptive_field = self.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.dataset2index = {}
                if mode == 'train' or mode== 'ori' or mode == 'test':
                    for i, data_graph in enumerate(dataset_use):
                        self.dataset2index[data_graph] = i
                        n_dataset = A_dict[data_graph].shape[0]
                        self.nodevec1_pretrain = nn.Parameter(torch.randn(n_dataset, 10).to(self.device),
                                             requires_grad=True).to(self.device)
                        self.nodevec2_pretrain = nn.Parameter(torch.randn(10, n_dataset).to(self.device),
                                             requires_grad=True).to(self.device)
                else:
                    for i, data_graph in enumerate([dataset_use]):
                        self.dataset2index[data_graph] = i
                        n_dataset = A_dict[data_graph].shape[0]
                        self.nodevec1_eval = nn.Parameter(torch.randn(n_dataset, 10).to(self.device),
                                            requires_grad=True).to(self.device)
                        self.nodevec2_eval = nn.Parameter(torch.randn(10, n_dataset).to(self.device),
                                             requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels, self.residual_channels,
                                          self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: '+str(self.receptive_field))

    def forward(self, source, select_dataset, adap=None):
        inputs = source
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)


        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            if self.mode == 'train' or self.mode == 'ori' or self.mode == 'test':
                adp = F.softmax(F.relu(torch.mm(self.nodevec1_pretrain,
                                                self.nodevec2_pretrain)), dim=1)
            else:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1_eval,
                                                self.nodevec2_eval)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            if self.gcn_bool and self.supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            else:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                x = self.residual_convs[i](x)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
        # mem aug
        x = F.relu(skip)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)
        # (batch_size, output_window, num_nodes, self.output_dim)
        return x

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"

