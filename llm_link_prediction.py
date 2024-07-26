from openai import OpenAI
import pandas as pd
import openai
import openai
import re
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score,  precision_score, recall_score, f1_score
import networkx as nx

from rag.rag_model import rag_chat
from preprocessing.small_disease_network import split_edges
from preprocessing.large_disease_network import split_edges

train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_edges(g)

os.environ["OPENAI_API_KEY"]  = 'sk-YOUR OPEN AI KEY'
client = OpenAI()
node = pd.read_csv("updated_disease_descriptions.csv")

output_templates = {
    'zero_shot': "Is there a potential relationship between {src_name} and {dst_name}? {src_name} is {src_dec}, {dst_name} is {dst_dec}. Evaluate and respond with '1' for a strong link and '0' for a weak or no link. Do Not provide your reasoning.",
    'few_shot': ("For example: There are links in the following diseases: Node ID: {train_src_id}, Disease: {train_src} has relationship with Node ID: {train_dst_id}, Disease: {train_dst}. "
                 "Is there a potential relationship between {src_name} and {dst_name}? {src_name} is {src_dec}, {dst_name} is {dst_dec}. Evaluate and respond with '1' for a strong link and '0' for a weak or no link. Do Not provide your reasoning."),
    'chain_of_thought': ("Step-by-step, analyze whether there is a potential relationship between {src_name} and {dst_name}. "
                         "That indicates one might lead to or be associated with the other. "
                         "Evaluate and respond with '1' for a strong link and '0' for a weak or no link. Do Not provide your reasoning."),
    'full_graph_analysis': ("In a disease network, Disease {src_name} has {src_degree} connections, Disease {dst_name} has {dst_degree} connections. "
                            "They share {common_count} common diseases. Is there a potential relationship between {src_name} and {dst_name}? "
                            "That indicates one might lead to or be associated with the other. "
                            "Evaluate and respond with '1' for a strong link and '0' for a weak or no link. Do Not provide your reasoning.")
}



def generate_description(graph, src, dst, df, output_type='zero_shot'):
    # Fetching node information
    src_name = df[df['Id'] == src]['label'].values[0]
    src_desc = df[df['Id'] == src]['Description'].values[0]
    dst_name = df[df['Id'] == dst]['label'].values[0]
    dst_desc = df[df['Id'] == dst]['Description'].values[0]
    
    # Calculating graph metrics
    common_neighbors = set(graph.successors(src)).intersection(set(graph.successors(dst)))
    common_neighbors_names = df[df['Id'].isin(common_neighbors)]['label'].tolist()
    src_degree = graph.out_degree(src)
    dst_degree = graph.out_degree(dst)
    path_exists = "Yes" if nx.has_path(graph, src, dst) else "No"
    shortest_path = nx.shortest_path_length(graph, src, dst) if path_exists == "Yes" else 'N/A'
    
    # Selecting the appropriate template
    template = output_templates[output_type]
    return template.format(
        src_name=src_name, src_desc=src_desc, dst_name=dst_name, dst_desc=dst_desc,
        common_count=len(common_neighbors), common_neighbors=', '.join(common_neighbors_names),
        src_degree=src_degree, dst_degree=dst_degree, path_exists=path_exists, shortest_path=shortest_path
    )

def compute_scores(graph, df, pos=True):
    """ Compute scores for all edges in a graph """
    src, dst = graph.edges()
    scores = []
    for s, d in zip(src.tolist(), dst.tolist()):
        desc = generate_description(graph, s, d, df)
        score = query_gpt(desc)
        if score is not None:  # Ensure score is not None
            scores.append(score)
    return np.array(scores)

def query_gpt(description):
    response = rag_chat({"query": description})
    text_response = response['result'].strip()
    #text_response = response.choices[0].message.content.strip()
    print(text_response)
    # Use regex to find the first occurrence of a numeric response
    match = re.search(r'\b[01]\b', text_response)
    if match:
        return int(match.group(0))  # Return the found binary number

    # Handling non-binary responses
    if "no" in text_response.lower() or "not" in text_response.lower():
        return 0  # Assuming a negative response implies '0'
    elif "yes" in text_response.lower():
        return 1  # Assuming a positive response implies '1'

    # Default or error handling case
    print(f"Unexpected response: {text_response}")
    return None  # Return None or consider raising an exception or logging an errorr


def print_first_ten_edges_with_disease(graph, graph_name, df):
    # Extract source and destination node IDs
    src, dst = graph.edges()

    # Convert to lists for easier handling
    src_list = src.tolist()
    dst_list = dst.tolist()

    # Print the first ten edges with disease terms
    print(f"There are links in the following dieases:")
    for i in range(min(10, len(src_list))):  # Ensure we do not go out of bounds
        src_id = src_list[i]
        dst_id = dst_list[i]
        src_name = df[df['Id'] == src_id]['Disease Term'].values[0]  # Lookup source disease term
        dst_name = df[df['Id'] == dst_id]['Disease Term'].values[0]  # Lookup destination disease term
        print(f"Node ID: {src_id}, Disease: {src_name} has relatiobship with Node ID: {dst_id}, Disease: {dst_name}")

# Example usage, assuming you have a DataFrame 'members_df' with 'Id' and 'Disease Term' columns
print_first_ten_edges_with_disease(train_pos_g, "Positive Training Graph", node)

def print_first_ten_neg_edges_with_disease(graph, graph_name, df):
    # Extract source and destination node IDs
    src, dst = graph.edges()

    # Convert to lists for easier handling
    src_list = src.tolist()
    dst_list = dst.tolist()

    # Print the first ten edges with disease terms
    print("There are no links in the following dieases:")
    for i in range(min(10, len(src_list))):  # Ensure we do not go out of bounds
        src_id = src_list[i]
        dst_id = dst_list[i]
        src_name = df[df['Id'] == src_id]['Disease Term'].values[0]  # Lookup source disease term
        dst_name = df[df['Id'] == dst_id]['Disease Term'].values[0]  # Lookup destination disease term
        print(f"Node ID: {src_id}, Disease: {src_name} has no relatiobship with Node ID: {dst_id}, Disease: {dst_name}")

# Example usage, assuming you have a DataFrame 'members_df' with 'Id' and 'Disease Term' columns
print_first_ten_neg_edges_with_disease(train_pos_g, "NEGATIVE Training Graph", node)


def compute_scores(graph, df, pos=True):
    """Compute scores and print predictions for all edges in a graph, ensuring binary responses."""
    src, dst = graph.edges()
    scores = []
    predictions = []  # List to store predictions
    for s, d in zip(src.tolist(), dst.tolist()):
        desc = generate_description(graph, s, d, df)
        score = query_gpt(desc)
        if score is not None and score in [0, 1]:  # Ensure score is strictly binary
            scores.append(score)
            prediction_text = f"{desc} Prediction: {score}"
        else:
            prediction_text = f"{desc} Prediction: Error - non-binary response received"

        predictions.append(prediction_text)
        print(prediction_text)  # Print each prediction

    # Optionally print all predictions for a summary
    if pos:
        print("Predictions on Positive Samples:")
    else:
        print("Predictions on Negative Samples:")
    for prediction in predictions:
        print(prediction)

    return np.array(scores), predictions  # Return scores and predictions


# Example usage (assuming you have these graphs initialized properly)
#pos_train_scores = compute_scores(train_pos_g)
#neg_train_scores = compute_scores(train_neg_g)
# Example usage (make sure to pass the member_df DataFrame)
# Example usage
pos_test_scores, pos_predictions = compute_scores(test_pos_g, node, pos=True)
neg_test_scores, neg_predictions = compute_scores(test_neg_g, node, pos=False)



# Compute AUC
test_auc = roc_auc_score(
    np.concatenate([np.ones(len(pos_test_scores)), np.zeros(len(neg_test_scores))]),
    np.concatenate([pos_test_scores, neg_test_scores])
)
print("Test AUC:", test_auc)

ap_score = average_precision_score(
    np.concatenate([np.ones(len(pos_test_scores)), np.zeros(len(neg_test_scores))]),
    np.concatenate([pos_test_scores, neg_test_scores]))
print("Average Precision Score:", ap_score)

threshold = 0.5  # Threshold for classifying scores as positive

# Concatenate all scores and labels
scores = np.concatenate([pos_test_scores, neg_test_scores])
labels = np.concatenate([np.ones(len(pos_test_scores)), np.zeros(len(neg_test_scores))])  # True labels

# Convert scores to binary predictions based on the threshold
predictions = (scores >= threshold).astype(int)

# Compute the F1 score
f1 = f1_score(labels, predictions)
print("F1 Score:", f1)

# Compute precision
precision = precision_score(labels, predictions)
print("Precision:", precision)

# Compute recall
recall = recall_score(labels, predictions)
print("Recall:", recall)
