import os
import json
import shutil
import torch
import numpy as np
from torch_geometric.data import Dataset, Data , HeteroData
from transformers import AutoTokenizer, ViTFeatureExtractor, BertTokenizer, CLIPProcessor
from tqdm import tqdm
from utils import preprocess, image_transforms, normalize

class DualGraphDataset(Dataset):
    def __init__(self, root, image_dir, split, classes, language='en', max_length=64, transform=None, pre_transform=None, aug = False):

        self.split = split
        self.filename = "{}.json".format(split)
        self.aug = aug
        self.classes = classes
        self.language = language
        self.root = root
        self.image_dir = image_dir
        self.max_length = max_length
        self.max_nodes = 100 if self.split == "train" else 80

        self.textTokenizer = self._get_tokenizer()

        super(DualGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        data_len = (len(self.data))
        return [f'data_{self.split}_{i}.pt' for i in range(data_len)]

    def download(self):
        download_path = self.raw_dir
        os.makedirs(download_path,exist_ok=True)
        file_path = os.path.join(self.root, self.filename)
        shutil.copy(file_path,download_path)

    def process(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        for index, tweet in (enumerate(tqdm(self.data))):

            tweet_id = tweet['id']
            tweet["node0s"] = tweet["node0s"][:self.max_nodes]
            tweet["edge0s"] = tweet["edge0s"][:self.max_nodes-1]

            tweet["node1s"] = tweet["node1s"][:self.max_nodes]
            tweet["edge1s"] = tweet["edge1s"][:self.max_nodes-1]

            tweet["node2s"] = tweet["node2s"][:self.max_nodes]
            tweet["edge2s"] = tweet["edge2s"][:self.max_nodes-1]

            node_feat0s = self._get_node_features(tweet["node0s"])
            node_feat1s = self._get_node_features(tweet["node1s"])
            node_feat2s = self._get_node_features(tweet["node2s"])

            edge_index0 = self._get_adjacency_info(tweet["edge0s"])
            edge_index1 = self._get_adjacency_info(tweet["edge1s"])
            edge_index2 = self._get_adjacency_info(tweet["edge2s"])

            label = self._get_labels(tweet['label'])

            data = HeteroData(                        
                        y=label,
                        id=tweet_id,
                        )
            data['x0'].x = node_feat0s
            data['x0'].edge_index = edge_index0
            data['x1'].x = node_feat1s
            data['x1'].edge_index = edge_index1
            data['x2'].x = node_feat2s
            data['x2'].edge_index = edge_index2

            torch.save(data,
                       os.path.join(self.processed_dir,
                                    f'data_{self.split}_{index}.pt'))

    def _get_tokenizer(self):
        if self.language == 'en':
            return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
        elif self.language == 'cn':
            return AutoTokenizer.from_pretrained("bert-base-chinese")

    def _get_node_features(self, nodes):
        texts = [preprocess(node['text']) for node in nodes]
        encoded_input = self.textTokenizer.batch_encode_plus(
            texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')

        all_node_feats = torch.stack([
            encoded_input["input_ids"], encoded_input["attention_mask"]], dim=-1)
        return all_node_feats

    def _get_edge_features(self, edge_len):
        return torch.ones(edge_len, 1)

    def _get_adjacency_info1(self, edges):
        edge_indices = []
        
        for edge in edges:
            i = int(edge['from'])
            j = int(edge['to'])
            edge_indices += [[j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices


    def _get_adjacency_info(self, edges):
        edge_indices = []
        
        for edge in edges:
            i = int(edge['from'])
            j = int(edge['to'])
            edge_indices += [[j, i]]

            while j != 0:
                for edge2 in edges:
                    edge_from = int(edge2['from'])
                    edge_to = int(edge2['to'])
                    if edge_from == j:
                        j = edge_to
                        edge_indices += [[j, i]]
                        continue

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices

    def _get_node_features2(self, nodes, edges):

        source = str(nodes[0]["text"])

        texts_from = [preprocess(nodes[edge['from']]['text']) for edge in edges]
        texts_to = [preprocess(nodes[edge['to']]['text']) for edge in edges]
        texts_from.insert(0,preprocess(source))
        texts_to.insert(0,"")

        encoding = self.textTokenizer(texts_from, texts_to, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')
        all_node_feats = torch.stack([
            encoding["input_ids"], encoding["attention_mask"]], dim=-1)

        return all_node_feats


    def _get_labels(self, label):
        label = self.classes.index(label)
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int32)

    def len(self):
        return len(self.data)

    def get(self, idx):

        data = torch.load(os.path.join(self.processed_dir,
                          f'data_{self.split}_{idx}.pt'))
    
        return data