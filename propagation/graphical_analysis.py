import queue

import numpy as np
from analysis_util import get_numpy_array, BaseFeatureHelper
from propagation.util.util import tweet_node


def get_page_rank(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    page_arr = []

    while q.qsize() != 0:
        node = q.get()
        if node.rank > 0:
            page_arr.append(node.rank)

    if len(page_arr) == 0:
        return 0
    else:
        return np.mean(np.array(page_arr))


def get_closeness(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    num_close_arr = []

    while q.qsize() != 0:
        node = q.get()
        num_close_arr.append(node.close_centre)

    if len(num_close_arr) == 0:
        return 0
    else:
        return np.mean(np.array(num_close_arr))


def get_edge_centre(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()
        sent_arr.append(node.edge_centre)

    if len(sent_arr) == 0:
        return 0
    else:
        return np.mean(np.array(sent_arr))


def get_strong_conn(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()
        if node.strong_conn > 0:
            sent_arr.append(node.strong_conn)

    if len(sent_arr) == 0:
        return 0
    else:
        return np.mean(np.array(sent_arr))


def get_attr_conn(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()
        if node.attr_conn > 0:
            sent_arr.append(node.attr_conn)

    if len(sent_arr) == 0:
        return 0
    else:
        return np.mean(np.array(sent_arr))


def get_branch_weight(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()
        if node.branch_weight > 0:
            sent_arr.append(node.branch_weight)

    if len(sent_arr) == 0:
        return 0
    else:
        return np.mean(np.array(sent_arr))


def get_stats_for_features(news_graps: list, get_feature_fun_ref, print=False, feature_name=None):
    result = []
    for graph in news_graps:
        result.append(get_feature_fun_ref(graph))

    if print:
        print_stat_values(feature_name, result)

    return result


def print_stat_values(feature_name, values):
    print("=========================================")
    print("Feature : {}".format(feature_name))
    print("Min value : {}".format(min(values)))
    print("Max value : {}".format(max(values)))
    print("Mean value : {}".format(np.mean(np.array(values))))
    print("=========================================")


def get_all_textual_features(prop_graphs, micro_features, macro_features):
    macro_features_functions = [get_page_rank,
                                get_closeness,
                                get_attr_conn,
                                get_strong_conn,
                                get_edge_centre,
                                get_branch_weight,
                                ]

    function_refs = []

    if macro_features:
        function_refs.extend(macro_features_functions)

    all_features = []

    for function_reference in function_refs:
        features_set = get_stats_for_features(prop_graphs, function_reference, print=False, feature_name=None)
        all_features.append(features_set)

    return np.transpose(get_numpy_array(all_features))


class GraphicalFeatureHelper(BaseFeatureHelper):

    def get_micro_feature_method_references(self):
        return []

    def get_micro_feature_method_names(self):
        return []

    def get_micro_feature_short_names(self):
        return []

    def get_feature_group_name(self):
        return "gr"

    def get_macro_feature_method_references(self):
        method_refs = [get_page_rank,
                       get_closeness,
                       get_attr_conn,
                       get_strong_conn,
                       get_edge_centre,
                       get_branch_weight,
                       ]

        return method_refs

    def get_macro_feature_method_names(self):
        feature_names = ["Average Page Rank",
                         "Average Closeness of the Graph",
                         "Average Attractive Components of the graph",
                         "Average Strong Connections of the Graph",
                         "Average Edge Centre of the Graph",
                         "Average Branch Weight of the Graph",
                         ]

        return feature_names

    def get_macro_feature_short_names(self):
        feature_names = []
        for fea in range(len(self.get_macro_feature_method_names())):
            feature_names.append("G" + str(fea+1))
        return feature_names
