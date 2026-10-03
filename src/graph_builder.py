import pandas as pd
import torch
import os
import itertools
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

def build_graph():
    print("Loading preprocessed datasets...")
    # Load all splits to construct the full graph (transductive learning setup)
    # We will use masks later during training to prevent label leakage.
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validate.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Add a column to identify the split
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine everything to create global node IDs
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df['node_id'] = full_df.index
    
    print(f"Total nodes in graph: {len(full_df)}")
    
    # Map Reddit 'id' string to our integer 'node_id'
    id_to_node = dict(zip(full_df['id'], full_df['node_id']))
    
    edges_src = []
    edges_dst = []
    
    print("Building edges based on linked_submission_id...")
    # 1. Edges from linked_submission_id
    # If a post replies to or links another submission, connect them.
    linked_count = 0
    for _, row in tqdm(full_df.iterrows(), total=len(full_df)):
        link_id = str(row['linked_submission_id'])
        if link_id != "" and link_id != "nan" and link_id in id_to_node:
            src = row['node_id']
            dst = id_to_node[link_id]
            edges_src.extend([src, dst])  # Undirected edge
            edges_dst.extend([dst, src])
            linked_count += 1
            
    print(f"Created {linked_count * 2} directed edges from linked submissions.")
    
    print("Building edges based on shared Authors...")
    # 2. Edges from same author
    # We connect posts made by the same author.
    # To prevent massive dense cliques that blow up RAM, we ignore "unknown_author"
    author_groups = full_df[full_df['author'] != 'unknown_author'].groupby('author')['node_id'].apply(list)
    
    author_count = 0
    for author, node_list in tqdm(author_groups.items(), total=len(author_groups)):
        if len(node_list) > 1 and len(node_list) <= 50:  # Cap at 50 posts per author to prevent memory explosion
            # Create edges between all combinations of this author's posts
            for u, v in itertools.combinations(node_list, 2):
                edges_src.extend([u, v])
                edges_dst.extend([v, u])
                author_count += 1
                
    print(f"Created {author_count * 2} directed edges from shared authors.")
    
    # Note: We intentionally DO NOT connect all posts in the same subreddit. 
    # Connecting 10,000 posts in a subreddit would create 100,000,000 edges just for one group!
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    print(f"Final Graph Shape: Nodes = {len(full_df)}, Edges = {edge_index.shape[1]}")
    
    # Save node mapping and edges
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save node metadata and masks
    full_df[['node_id', 'id', 'split', 'label', 'clean_title']].to_csv(os.path.join(OUTPUT_DIR, "node_mapping.csv"), index=False)
    
    # Save edge index
    torch.save(edge_index, os.path.join(OUTPUT_DIR, "edge_index.pt"))
    print(f"Saved graph structures to {OUTPUT_DIR}")

if __name__ == "__main__":
    build_graph()
