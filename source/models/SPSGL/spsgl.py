import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from .components import brain_pathway_transformer
from omegaconf import DictConfig
from ..base import BaseModel
from torch_geometric.utils import to_dense_adj
from .bmt import brain_message_transmission
brain_networks = {
    "norm":[],
    "DMN": [5, 6, 7, 8, 9, 10, 27, 28, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 61, 62, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    "FPN": [7, 8, 11, 12, 13, 14, 31, 32, 35, 36, 61, 62, 63, 64, 67, 68, 71, 72, 77, 78, 85, 86, 89, 90],
    "SN":  [19, 20, 21, 22, 29, 30, 31, 32, 39, 40, 41, 42, 71, 72, 73, 74, 77, 78]
}

def create_symmetric_adjacency_matrix(edge_index, batch_idx, edge_attr, num_roi):
    enhanced_fc = to_dense_adj(
        edge_index=edge_index,
        batch=batch_idx,
        edge_attr=edge_attr.squeeze()  # 移除可能的单维度
    )
    bs = enhanced_fc.size(0)
    enhanced_fc = enhanced_fc.view(bs, num_roi, num_roi)
    enhanced_fc = (enhanced_fc + enhanced_fc.transpose(1, 2))
    enhanced_fc.diagonal(dim1=1, dim2=2).div_(2)

    return enhanced_fc
class TransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.brain_networks = brain_networks
        self.brain_weight = torch.tensor([0.0, 0.4, 0.8], dtype=torch.float32)
        self.num_head = 4
        self.dim = input_feature_size
        brain_structure = self.construct_brain_structure()
        self.weight_matrix = self.get_weight_matrix(brain_structure)
        self.bpt = brain_pathway_transformer(d_model=input_feature_size, nhead=self.num_head,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def construct_brain_structure(self):
        brain_networks = self.brain_networks
        num_head = self.num_head
        keys = list(brain_networks.keys())
        if len(keys) > num_head:
            selected_brain_networks = {k: brain_networks[k] for k in keys[:num_head]}
        else:
            selected_brain_networks = brain_networks.copy()
            index = 0
            while len(selected_brain_networks) < num_head:
                key_to_copy = keys[index % len(keys)]
                new_key = f"{key_to_copy}_{len(selected_brain_networks)}"
                selected_brain_networks[new_key] = brain_networks[key_to_copy]
                index += 1
        num_brains = len(selected_brain_networks)
        f = torch.zeros((num_brains, self.dim, self.dim), dtype=torch.int32)
        for i, brain_network in enumerate(selected_brain_networks):
            i_brain = selected_brain_networks[brain_network]
            if i_brain and min(i_brain) > 0:
                i_brain = torch.tensor([xx - 1 for xx in i_brain], dtype=torch.long)  # 转换为 Tensor
            if len(i_brain) > 0:
                f[i, i_brain[:, None], i_brain] = 2
                row_mask = f[i, i_brain] == 0  # 仅当值为 0 时，才能被赋值 1
                col_mask = f[i, :, i_brain] == 0
                f[i, i_brain] = torch.where(row_mask, 1, f[i, i_brain])
                f[i, :, i_brain] = torch.where(col_mask, 1, f[i, :, i_brain])
        for i in range(num_brains):
            f[i].diagonal().zero_()
        return f

    def get_weight_matrix(self,brain_structure):
        brain_structure = brain_structure.to(self.brain_weight.device)
        f = torch.where(brain_structure == 2, self.brain_weight[2], brain_structure)
        f = torch.where(f == 1, self.brain_weight[1], f)
        f = torch.where(f == 0, self.brain_weight[0], f)
        return f
    def forward(self, x):
        x = self.bpt(x,weight_matrix = self.weight_matrix)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.bpt.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class SPSGL(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz
        self.num_roi = forward_dim
        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        self.bmt = brain_message_transmission(node_feat_dim=4, edge_feat_dim=1, hidden_dim=32)
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
                bacth_graph):
        x = bacth_graph.x
        edge_index = bacth_graph.edge_index
        edge_attr = bacth_graph.edge_attr
        batch_idx = bacth_graph.batch
        updated_edge_attr= self.bmt(x, edge_index, edge_attr)
        fc_data = create_symmetric_adjacency_matrix(edge_index = edge_index, batch_idx=batch_idx, edge_attr=updated_edge_attr, num_roi=self.num_roi)
        edge_attr = fc_data
        bz, _, _, = fc_data.shape
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            fc_data = torch.cat([fc_data, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            fc_data, assignment = atten(fc_data)
            assignments.append(assignment)
        fc_data = self.dim_reduction(fc_data)
        fc_data = fc_data.reshape((bz, -1))
        return self.fc(fc_data),edge_attr

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None
        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
