import os
import pandas as pd
import numpy as np
import random
import networkx as nx
import seaborn as sns
from pathlib import Path
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyvis.network import Network
from helpers.df_helpers import documents2Dataframe, df2Graph, graph2Df
import gradio as gr
import logging

# Constants
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
WEIGHT_MULTIPLIER = 4
COLOR_PALETTE = "hls"
GRAPH_OUTPUT_DIRECTORY = "./docs/index.html"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def colors2Community(communities) -> pd.DataFrame:
    palette = sns.color_palette(COLOR_PALETTE, len(communities)).as_hex()
    random.shuffle(palette)
    rows = [{"node": node, "color": color, "group": group + 1} 
            for group, community in enumerate(communities) 
            for node, color in zip(community, palette)]
    return pd.DataFrame(rows)

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    dfg_long = pd.melt(df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node").drop(columns=["variable"])
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    dfg_wide = dfg_wide[dfg_wide["node_1"] != dfg_wide["node_2"]].reset_index(drop=True)
    dfg2 = dfg_wide.groupby(["node_1", "node_2"]).agg({"chunk_id": [",".join, "count"]}).reset_index()
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2

def load_documents(input_dir):
    loader = DirectoryLoader(input_dir, show_progress=True)
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, is_separator_regex=False)
    return splitter.split_documents(documents)

def save_dataframes(df, dfg1, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dfg1.to_csv(output_dir / "graph.csv", sep="|", index=False)
    df.to_csv(output_dir / "chunks.csv", sep="|", index=False)

def load_dataframes(output_dir):
    df = pd.read_csv(output_dir / "chunks.csv", sep="|")
    dfg1 = pd.read_csv(output_dir / "graph.csv", sep="|")
    return df, dfg1

def build_graph(dfg):
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for _, row in dfg.iterrows():
        G.add_edge(row["node_1"], row["node_2"], title=row["edge"], weight=row['count'] / WEIGHT_MULTIPLIER)
    return G

def visualize_graph(G, communities):
    colors = colors2Community(communities)
    for _, row in colors.iterrows():
        G.nodes[row['node']].update(group=row['group'], color=row['color'], size=G.degree[row['node']])
    nt = Network(notebook=False, cdn_resources="remote", height="900px", width="100%", select_menu=True)
    nt.from_nx(G)
    nt.force_atlas_2based(central_gravity=0.015, gravity=-31)
    nt.show_buttons(filter_=["physics"])
    html = nt.generate_html().replace("'", "\"")
    return f"""<iframe style="width: 100%; height: 600px; margin:0 auto" 
               name="result" allow="midi; geolocation; microphone; camera; 
               display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
               allow-scripts allow-same-origin allow-popups 
               allow-top-navigation-by-user-activation allow-downloads" allowfullscreen 
               allowpaymentrequest frameborder="0" srcdoc='{html}'></iframe>"""

def process_pdfs(input_dir, output_dir, regenerate=False):
    if regenerate:
        documents = load_documents(input_dir)
        pages = split_documents(documents)
        df = documents2Dataframe(pages)
        concepts_list = df2Graph(df, model='zephyr:latest')
        dfg1 = graph2Df(concepts_list)
        save_dataframes(df, dfg1, output_dir)
    else:
        df, dfg1 = load_dataframes(output_dir)
    
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = WEIGHT_MULTIPLIER
    dfg2 = contextual_proximity(dfg1)
    dfg = pd.concat([dfg1, dfg2], axis=0).groupby(["node_1", "node_2"]).agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'}).reset_index()
    G = build_graph(dfg)
    
    communities_generator = nx.community.girvan_newman(G)
    next_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)  # Two levels of communities
    communities = sorted(map(sorted, next_level_communities))
    logger.info(f"Number of Communities = {len(communities)}")
    logger.info(communities)
    
    html = visualize_graph(G, communities)
    return html

def main():
    data_dir = "cureus"
    input_dir = Path(f"./data_input/{data_dir}")
    output_dir = Path(f"./data_output/{data_dir}")
    html = process_pdfs(input_dir, output_dir, regenerate=False)
    
    demo = gr.Interface(fn=lambda: html, inputs=None, outputs=gr.HTML(), title="Text to knowledge graph", allow_flagging='never')
    demo.launch()

if __name__ == "__main__":
    main()
