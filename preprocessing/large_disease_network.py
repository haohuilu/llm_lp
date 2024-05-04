import itertools
import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from openai import OpenAI


df = pd.read_csv("DN.csv")
members = pd.read_csv("updated_disease_descriptions.csv")


# Get all unique disease terms from both columns
unique_terms = pd.unique(df[['MeSH Disease Term', 'MeSH Disease Term.1']].values.ravel('K'))

# Create a mapping from disease terms to integers
term_to_label = {term: i for i, term in enumerate(unique_terms)}

# Apply the mapping to both columns
df['MeSH Disease Term Label'] = df['MeSH Disease Term'].map(term_to_label)
df['MeSH Disease Term.1 Label'] = df['MeSH Disease Term.1'].map(term_to_label)

os.environ["OPENAI_API_KEY"]  = 'sk-YOUR OPEN AI KEY'
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

members['ada_embedding'] = members.Description.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
#members.to_csv('disease_embedding.csv', index=False)

interactions = df[["MeSH Disease Term Label", "MeSH Disease Term.1 Label", "symptom similarity score"]]

rename_mapping = {
    'MeSH Disease Term Label': 'Src',
    'MeSH Disease Term.1 Label': 'Dst',
    'symptom similarity score': 'Weight'
}

# Rename the columns using the mapping
interactions.rename(columns=rename_mapping, inplace=True)
interactions

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset


class DiseaseNetwork(DGLDataset):
    def __init__(self):
        super().__init__(name="disease_network")

    def process(self):
        nodes_data = members
        edges_data = interactions
       # node_features = np.stack(nodes_data['embeddings'].values)
      #  node_features = torch.tensor(node_features, dtype=torch.float32)
        node_labels = torch.from_numpy(
            nodes_data["Disease Term"].astype("category").cat.codes.to_numpy()
        )
        edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
   #     self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = edge_features


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


dataset = DiseaseNetwork()
g = dataset[0]

print(g)

def split_edges(g, test_ratio=0.01):
    # Assuming g is your graph and it has a method num_edges() and num_nodes()
    u, v = g.edges()  # Example edge lists, need to be tensors or arrays

    # Random permutation of edges
    eids = np.arange(g.num_edges())
    eids = np.random.permutation(g.num_edges())
    test_size = int(len(eids) * test_ratio)
    train_size = g.num_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Creating a square adjacency matrix
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.num_nodes(), g.num_nodes()))

    # Subtracting the identity matrix
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())

    # Finding non-zero elements (negative edges)
    neg_u, neg_v = np.where(adj_neg != 0)

    # Random choice of negative edges, ensuring the number matches the number of edges in the graph
    neg_eids = np.random.choice(len(neg_u), g.num_edges())

    # Splitting negative edges into training and testing sets
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

# Usage:
train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_edges(g)