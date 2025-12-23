import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class brain_message_transmission(MessagePassing):

    def __init__(self, node_feat_dim=4, edge_feat_dim=1, hidden_dim=32):
        super().__init__(aggr='add')
        self.slow_band_names = ['slow3', 'slow4', 'slow5', 'slow3-5']
        self.reset_gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * node_feat_dim + edge_feat_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
            ) for name in self.slow_band_names
        })
        self.update_gate = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_net = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, edge_feat_dim),
            nn.Tanh()
        )
        self.training_step = 0

    def forward(self, x, edge_index, edge_attr):
        self.training_step += 1
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return inputs

    def get_update_gate_input(self, x_i, x_j, edge_attr):
        return torch.abs(x_i - x_j)

    def get_edge_net_input(self, weighted_x, edge_attr):
        return torch.cat([weighted_x, edge_attr], dim=1)

    def message(self, x_i, x_j, edge_attr):
        input_cat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        reset_signals = torch.stack([
            gate(input_cat) for gate in self.reset_gates.values()
        ], dim=1).squeeze(-1)
        reset_signals = F.softmax(reset_signals, dim=1)
        weighted_x = 0.5 * (reset_signals * x_i + reset_signals * x_j)
        candidate = self.edge_net(self.get_edge_net_input(weighted_x, edge_attr))
        update_weight = self.update_gate(self.get_update_gate_input(x_i, x_j, edge_attr))
        return torch.cat([candidate, update_weight], dim=1)

    def update(self, aggr_msg, edge_attr):
        candidate = aggr_msg[:, 0:1]
        update_weight = aggr_msg[:, 1:2]
        return (1 - update_weight) * edge_attr + update_weight * candidate
