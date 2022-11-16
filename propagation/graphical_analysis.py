import queue

import numpy as np
from analysis_util import get_numpy_array, BaseFeatureHelper
from propagation.util.util import tweet_node


def get_page_rank(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    len_tweet_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None and len(tweet) > 0:
                len_tweet_arr.append(len(tweet))

    if len(len_tweet_arr) == 0:
        return 0
    else:
        return np.mean(np.array(len_tweet_arr))


def get_effectiveness_size(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    num_url_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                num_url_arr.append({})

    if len(num_url_arr) == 0:
        return 0
    else:
        return np.mean(np.array(num_url_arr))


def get_closeness(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    num_hastags_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                tags = {tag.strip("#") for tag in tweet.split() if tag.startswith("#")}
                if len(tags) > 0:
                    num_hastags_arr.append(len(tags))

    if len(num_hastags_arr) == 0:
        return 0
    else:
        return np.mean(np.array(num_hastags_arr))


def get_eigen_centre(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                sent_arr.append({})
                break
            else:
                break

    if len(sent_arr) == 0:
        return 0
    else:
        emo_attr_dict = dict.fromkeys(sent_arr[0].keys(), [])

        for sent in sent_arr:
            for key, value in sent.items():
                arr_val = emo_attr_dict[key].copy()
                arr_val.append(value)
                emo_attr_dict[key] = arr_val

        averages = [(k, sum(v) / len(v)) for k, v in emo_attr_dict.items()]
        emo_mean_dict = {}
        for k, v in averages:
            emo_mean_dict[k] = v
        return emo_mean_dict


def get_edge_centre(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                sent_arr.append({})

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
        news_title = node.news_title
        news_text = node.news_text
        text_to_cons = None
        try:
            if news_title is not None and len(news_title) > 0:
                text_to_cons = news_title
            elif news_text is not None and len(news_text) > 0:
                text_to_cons = news_text
        except:
            print(news_title)
            print("/////////////")
            print(news_text)
            if news_text is not None and len(news_text) > 0:
                text_to_cons = news_text

        if text_to_cons is not None and len(text_to_cons) > 0:
            for child in node.children:
                q.put(child)
                tweet = child.text
                if tweet is not None and len(tweet) > 0:
                    sent_arr.append({})

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

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                sent_arr.append({})

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

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                sent_dict = []
                sent_arr.append(sent_dict)

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
                                get_effectiveness_size,
                                get_eigen_centre,
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
        return "tex"

    def get_macro_feature_method_references(self):
        method_refs = [get_page_rank,
                       get_closeness,
                       get_attr_conn,
                       get_strong_conn,
                       get_edge_centre,
                       get_branch_weight,
                       get_effectiveness_size,
                       get_eigen_centre,
                       ]

        return method_refs

    def get_macro_feature_method_names(self):
        feature_names = ["Length of the tweet",
                         "Count of the URL",
                         "Emotion behind the tweet",
                         "Closeness of the tweet with the news title in macro network",
                         "Number of hash tags in tweet in macro network",
                         "Score of positive in tweet in macro network",
                         "Score of negative in tweet in macro network",
                         "Sentiment score of the tweet in macro network",
                         "Number of people mentioned in macro network"
                         ]

        return feature_names

    def get_macro_feature_short_names(self):
        feature_names = []
        for fea in range(len(self.get_macro_feature_method_names())):
            feature_names.append("G" + str(fea))
        return feature_names
