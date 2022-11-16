import json
import os
import pickle
import sys
import pandas as pd
import ast
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm

from propagation.util.constants import RETWEET_NODE, REPLY_NODE
from propagation.util.util import tweet_node
from time import sleep
from random import random
from multiprocessing import Process

def construct_tweet_node_from_json(json_data):
    new_graph = json_graph.tree_graph(json_data)
    # item_temp = nx.DiGraph(new_graph).degree

    # itemps = nx.DiGraph.in_degree(new_graph).items
    itemps1 = nx.DiGraph(new_graph).in_degree()  # ignore warning

    root_node = [node for node, in_degree in itemps1 if in_degree == 0][0]
    node_id_obj_dict = dict()
    dfs_node_construction_helper(root_node, new_graph, set(), node_id_obj_dict)
    return node_id_obj_dict[root_node]


def dfs_node_construction_helper(node_id, graph: nx.DiGraph, visited: set, node_id_obj_dict: dict):
    if node_id in visited:
        return None

    visited.add(node_id)

    tweet_node_obj = construct_tweet_node_from_nx_node(node_id, graph)

    node_id_obj_dict[node_id] = tweet_node_obj

    for neighbor_node_id in graph.successors(node_id):
        if neighbor_node_id not in visited:
            dfs_node_construction_helper(neighbor_node_id, graph, visited, node_id_obj_dict)
            add_node_object_edge(node_id, neighbor_node_id, node_id_obj_dict)


def add_node_object_edge(parent_node_id: int, child_node_id: int, node_id_obj_dict: dict):
    parent_node = node_id_obj_dict[parent_node_id]
    child_node = node_id_obj_dict[child_node_id]

    if child_node.node_type == RETWEET_NODE:
        parent_node.add_retweet_child(child_node)
    elif child_node.node_type == REPLY_NODE:
        parent_node.add_reply_child(child_node)
    else:
        # news node add both retweet and reply edge
        parent_node.add_retweet_child(child_node)
        parent_node.add_reply_child(child_node)


def construct_tweet_node_from_nx_node(node_id, graph: nx.DiGraph):
    tweet_node_model = tweet_node(tweet_id=graph.nodes[node_id]['tweet_id'],
                                  created_time=graph.nodes[node_id]['time'],
                                  node_type=graph.nodes[node_id]['type'],
                                  user_id=graph.nodes[node_id]['user'],
                                  botometer_score=graph.nodes[node_id].get('bot_score', None),
                                  sentiment=graph.nodes[node_id].get('sentiment', None),
                                  text=graph.nodes[node_id].get('tweet', None),
                                  news_title=graph.nodes[node_id].get('title', None),
                                  news_text=graph.nodes[node_id].get('text', None),
                                  )

    # if tweet_node_model.node_type == 1:
    #     page_rank = nx.pagerank(graph, alpha=0.9)
    #     page_rank_arr = np.mean(np.array(list(page_rank.values())))
    #
    #     close_centre = nx.closeness_centrality(graph, u=None, distance=None, wf_improved=True)
    #     close_centre_arr = np.mean(np.array(list(close_centre.values())))
    #
    #     constraint = nx.constraint(graph, nodes=None)
    #     constraint_arr = np.mean(np.array(np.nan_to_num(list(constraint.values()))))
    #
    #     edge_centre = nx.edge_betweenness_centrality(graph, k=None, normalized=True, weight=None, seed=None)
    #     edge_centre_arr = np.mean(np.array(list(edge_centre.values())))
    #
    #     effective_size = nx.effective_size(graph, nodes=None)
    #
    #     effective_size_arr = np.mean(np.array(np.nan_to_num(list(effective_size.values()))))
    #
    #     strong_conn = nx.number_strongly_connected_components(graph)
    #     attr_conn = nx.number_attracting_components(graph)
    #     branch_weight = nx.tree.branching_weight(graph)
    #
    #     tweet_node_model.rank = page_rank_arr
    #     tweet_node_model.close_centre = close_centre_arr
    #     tweet_node_model.constraint = constraint_arr
    #     tweet_node_model.edge_centre = edge_centre_arr
    #     tweet_node_model.effective_size_score = effective_size_arr
    #     tweet_node_model.strong_conn = strong_conn
    #     tweet_node_model.attr_conn = attr_conn
    #     tweet_node_model.branch_weight = branch_weight

    return tweet_node_model


def get_dataset_sample_ids(news_source, news_label, dataset_dir="data/sample_ids"):
    sample_list = []
    if news_source == "politifact":
        dataset_dir = sys.path[4] + "/"
        df = pd.read_csv(dataset_dir + "politifact_prop.csv")
        if news_label == "fake":
            sample_list = df[df["target"] == 0]["id"].tolist()
        else:
            sample_list = df[df["target"] == 1]["id"].tolist()

    return sample_list


def load_from_nx_graphs(dataset_dir: str, news_source: str, news_label: str):
    tweet_node_objects = []

    news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
    # sample_ids = get_dataset_sample_ids(news_source, news_label, "data/sample_ids")
    tweet_list = []
    if news_source == "politifact":
        dataset_dir = sys.path[4] + "/pickle/merged_tweet_df.pkl"
        # df = pd.read_csv(dataset_dir + "politifact_prop.csv")
        df = pickle.load(open(dataset_dir, "rb"))
        if news_label == "fake":
            tweet_list = df[df["target"] == 0]["tweet_mod"].tolist()
        else:
            tweet_list = df[df["target"] == 1]["tweet_mod"].tolist()

    # processes = [Process(target=convert_to_tweet_node_obj(tweet_list[i]), args=(i,)) for i in range(len(tweet_list))]
    for tweet in tweet_list:
        # tweet_dict = tweet[0]
        tweet_node_conv = convert_to_tweet_node_obj(tweet)
        tweet_node_objects.append(tweet_node_conv)

    # for process in processes:
    #     process.start()
    # # wait for all processes to complete
    # for process in processes:
    #     process.join()
    #     # report that all tasks are completed
    # print('Done', flush=True)

    return tweet_node_objects


def convert_to_tweet_node_obj(tweet):
    tweet_dict = tweet[0]
    tweet_node_obj = construct_tweet_node_from_json(tweet_dict)
    return tweet_node_obj


def load_networkx_graphs(dataset_dir: str, news_source: str, news_label: str):
    news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)

    news_samples = []

    for news_file in os.listdir(news_dataset_dir):
        with open("{}/{}".format(news_dataset_dir, news_file)) as file:
            news_samples.append(json_graph.tree_graph(json.load(file)))

    return news_samples


def load_networkx_graphs_from_df(dataset_dir: str, news_source: str, news_label: str):
    news_samples = []

    df = pd.read_csv(dataset_dir + "politifact_prop.csv")
    tweets = []

    if news_label == "real":
        tweets = df[df["target"] == 1]["tweet"].tolist()
    else:
        tweets = df[df["target"] == 0]["tweet"].tolist()

    for news_file in tweets:
        news_samples.append(json_graph.tree_graph(ast.literal_eval(news_file)))

    return news_samples


def load_dataset(dataset_dir: str, news_source: str):
    fake_news_samples = load_networkx_graphs_from_df(dataset_dir, news_source, "fake")
    real_news_samples = load_networkx_graphs_from_df(dataset_dir, news_source, "real")

    return fake_news_samples, real_news_samples


if __name__ == '__main__':
    print(sys.path[4])
    path = sys.path[4] + "/"
    fake_samples, real_samples = load_dataset(path, "politifact")
    all_graph = fake_samples + real_samples
    pickle.dump(all_graph, open(path + "all_graph_data", "wb"))
