import ast
import networkx as nx
import scipy.stats as sp
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import mpmath
from itertools import product
import os
import pandas as pd
import openai
import dgl
import torch
from dgl.data import DGLDataset
os.environ["DGLBACKEND"] = "pytorch"

# Read GML graph
G = ig.Graph.Read_GML('./data/human-disease.gml')

# Create a DataFrame from vertex attributes
node = pd.DataFrame({
    attribute: G.vs[attribute] for attribute in G.vertex_attributes()
})
node = node[['name', 'label']]

# Create a DataFrame from edge attributes
edges_df = pd.DataFrame({
    attribute: G.es[attribute] for attribute in G.edge_attributes()
})

# Include source and target information from the graph
edges_df['src'] = [G.vs[edge.source]['name'] if 'name' in G.vertex_attributes() else edge.source for edge in G.es]
edges_df['dsc'] = [G.vs[edge.target]['name'] if 'name' in G.vertex_attributes() else edge.target for edge in G.es]

# Get disease feature
os.environ["OPENAI_API_KEY"]  = 'sk-YOUROPENKEY'

client = OpenAI()
# System prompt for the AI
system_prompt = "You are an expert in the healthcare industry. Please provide a description of the following diseases."

# Function to create user prompt for each QA pair
def create_user_prompt(disease):
    return f"""
    Please provide a concise description of the following diseases, ideally in one sentence. Don't repeat the disease name in the answer.
    Disease: {disease}
    """

# Loop through each QA pair in the DataFrame
for index, row in node.iterrows():
    user_prompt = create_user_prompt(row['label'])

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    params = {
        "model": "gpt-4",
        "messages": prompt_messages,
        "max_tokens": 1024,
    }

    result = client.chat.completions.create(**params)
    description = result.choices[0].message.content.strip()
    node.at[index, 'Description'] = description  # Storing the response in the DataFrame
    print(prompt_messages)
    #print(description)  # Optional: print the response


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

node['ada_embedding'] = node.Description.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

edges_df["src"] = pd.to_numeric(edges_df["src"], errors='coerce')
edges_df["dsc"] = pd.to_numeric(edges_df["dsc"], errors='coerce')

# Assuming edges_data is your DataFrame with columns 'src' and 'dst' containing the node IDs
all_nodes = np.union1d(edges_df['src'].values, edges_df['dsc'].values)
node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}

edges_df['src'] = edges_df['src'].map(node_mapping)
edges_df['dsc'] = edges_df['dsc'].map(node_mapping)

# Create the graph using the newly mapped node IDs
g = dgl.graph((edges_df['src'].values, edges_df['dsc'].values), num_nodes=len(all_nodes))

# Create DGL graph

class SmallDiseaseNetwork(DGLDataset):
    def __init__(self):
        super().__init__(name="disease_network")

    def process(self):
        nodes_data = node
        edges_data = edges_df
       # node_features = np.stack(nodes_data['embeddings'].values)
      #  node_features = torch.tensor(node_features, dtype=torch.float32)
        node_labels = torch.from_numpy(
            nodes_data["label"].astype("category").cat.codes.to_numpy()
        )
        edges_src = torch.from_numpy(edges_data["src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["dsc"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
   #     self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


dataset = SmallDiseaseNetwork()
g = dataset[0]



# Assuming g is your graph and it has a method num_edges() and num_nodes()
u, v = g.edges()  # Example edge lists, need to be tensors or arrays

# Random permutation of edges
eids = np.arange(g.num_edges())
eids = np.random.permutation(g.num_edges())
test_size = int(len(eids) * 0.01)
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

all_nodes = node['name'].unique()
node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
node['Id'] = node['name'].map(node_mapping)

node.to_csv("small_disease_network_node.csv", index= None)
edges_df.to_csv("small_disease_network_edge.csv", index= None)