from torch_geometric.nn import global_add_pool,GATConv, GCNConv, TransformerConv, SAGEConv, global_max_pool, global_mean_pool, LayerNorm
from torch_geometric.nn.models import GraphSAGE, GAT
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, BertModel , RobertaModel
from transformers import ViTModel
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_sparse import SparseTensor
from utils import length_to_mask

class DualModel(nn.Module):
    def __init__(self, hidden_dim=768, label_dim=3, dropout_rate=0.1, language='en',mh_size = 2):
        super(DualModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language
        self.label_dim = label_dim
        self.mh_size = mh_size

        self.e1 = 'x0'
        self.e2 = 'x1'

        self.textEncoder = self.get_text_model()
        self.freeze_textEncoder()

        self.transformer1 = TransformerEncoder(TransformerEncoderLayer(hidden_dim, self.mh_size, hidden_dim, batch_first=True,dropout=dropout_rate), 2)
        self.transformer2 = TransformerEncoder(TransformerEncoderLayer(hidden_dim, self.mh_size, hidden_dim, batch_first=True,dropout=dropout_rate), 2)

        self.attn1 = nn.MultiheadAttention(hidden_dim, 1, batch_first=True,dropout=dropout_rate)
        self.attn2 = nn.MultiheadAttention(hidden_dim, 1, batch_first=True,dropout=dropout_rate)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc = nn.Linear(hidden_dim*2,label_dim)

    def get_text_model(self):
        if self.language == 'en':
            return AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        elif self.language == 'cn':
            return AutoModel.from_pretrained("bert-base-chinese")

    def freeze_textEncoder(self):
        for name, param in list(self.textEncoder.named_parameters()):
            if self.language == 'en':
                if 'pooler' in name or 'encoder.layer.11' in name :
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def pad(self, tensor, batch_index):
        num_seq = torch.unique(batch_index)
        tensors = [tensor[batch_index == seq_id] for seq_id in num_seq]
        lengths = [len(tensor) for tensor in tensors]
        lengths = torch.tensor(lengths).to(num_seq.device)
        masks = length_to_mask(lengths)
        return pad_sequence(tensors, batch_first=True), masks.bool()
    
    def get_x(self, x):
        x_id = x[:, :, 0].int()
        x_mask = x[:, :, 1]
        x = self.textEncoder(input_ids=x_id, attention_mask=x_mask)
        x = x[1]
        return self.dropout(x)

    def mean_pooling(self, x, x_mask):
        x_score = 1.0 / torch.sum((~x_mask).int().float(),1).unsqueeze(-1).expand(-1, x.size(1))
        x_score[x_mask] = float(0.0)
        x = x * x_score.unsqueeze(-1)
        x = torch.sum(x,1)
        return x

    def max_pooling(self,x, x_mask):
        x[x_mask] = float("-inf")
        x = torch.max(x,1)[0]
        return x

    def forward(self, data):

        x1, edge_index1, batch_index1 = data[self.e1].x, data[self.e1].edge_index, data[self.e1].batch

        x2, edge_index2, batch_index2 = data[self.e2].x, data[self.e2].edge_index, data[self.e2].batch

        # Feature Extraction
        x1 = self.get_x(x1)
        x2 = self.get_x(x2)
        x1 , x_mask1 = self.pad(x1,batch_index1)
        x2 , x_mask2 = self.pad(x2,batch_index2)

        # Missing Evidence
        # x1 = x1[:,0].unsqueeze(1)
        # x_mask1 = x_mask1[:,0].unsqueeze(1)
        # x2 = x2[:,0].unsqueeze(1)
        # x_mask2 = x_mask2[:,0].unsqueeze(1)

        # Self Attention
        x1 = self.transformer1(x1,src_key_padding_mask=x_mask1)
        s1 = x1[:,0].unsqueeze(1)
        x2 = self.transformer2(x2,src_key_padding_mask=x_mask2)
        s2 = x2[:,0].unsqueeze(1)

        # Cross Attention
        f1 = self.attn1(s1,x2,x2,key_padding_mask=x_mask2)[0]
        f2 = self.attn2(s2,x1,x1,key_padding_mask=x_mask1)[0]
        s1 = f1 + s1
        s2 = f2 + s2

        # Classification
        x = torch.cat([s1,s2],dim=-1).squeeze(1)
        x = self.fc(x)

        return x