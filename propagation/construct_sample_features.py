import pickle
import queue
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from analysis_util import equal_samples, get_sample_feature_value, get_numpy_array
from linguistic_analysis import get_all_linguistic_features, LinguisticFeatureHelper
from load_dataset import load_from_nx_graphs, load_from_nx_graphs_mix
from propagation.graphical_analysis import GraphicalFeatureHelper
from propagation.textual_analysis import TextualFeatureHelper
from structure_temp_analysis import get_all_structural_features, StructureFeatureHelper, get_first_post_time
from temporal_analysis import get_all_temporal_features, TemporalFeatureHelper
from propagation.util.util import tweet_node


def get_features(news_graphs, micro_features, macro_features):
    temporal_features = get_all_temporal_features(news_graphs, micro_features, macro_features)
    structural_features = get_all_structural_features(news_graphs, micro_features, macro_features)
    linguistic_features = get_all_linguistic_features(news_graphs, micro_features, macro_features)

    sample_features = np.concatenate([temporal_features, structural_features, linguistic_features], axis=1)
    return sample_features


def get_dataset(news_source, load_dataset=False, micro_features=True, macro_features=True):
    if load_dataset:
        sample_features = pickle.load(open("{}_samples_features.pkl".format(news_source), "rb"))
        target_labels = pickle.load(open("{}_target_labels.pkl".format(news_source), "rb"))

    else:
        fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/nx_network_data", news_source)

        # fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

        print("fake samples len : {} real samples len : {}".format(len(fake_prop_graph), len(real_prop_graph)))

        fake_news_samples = get_features(fake_prop_graph, micro_features, macro_features)
        real_news_samples = get_features(real_prop_graph, micro_features, macro_features)

        print("Fake feature array ")
        print(fake_news_samples.shape)

        print("real feature array")
        print(real_news_samples.shape)

        sample_features = np.concatenate([fake_news_samples, real_news_samples], axis=0)
        target_labels = np.concatenate([np.ones(len(fake_news_samples)), np.zeros(len(real_news_samples))], axis=0)

        pickle.dump(sample_features, (open("{}_samples_features.pkl".format(news_source), "wb")))
        pickle.dump(target_labels, (open("{}_target_labels.pkl".format(news_source), "wb")))

    return sample_features, target_labels


def get_train_test_split(samples_features, target_labels):
    X_train, X_test, y_train, y_test = train_test_split(samples_features, target_labels,
                                                        test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test


def perform_pca(train_data, target_labels):
    pca = PCA(n_components=min(20, len(train_data[0])))
    pca.fit(train_data, target_labels)
    return pca


def get_dataset_file_name(file_dir, news_source, include_micro=True, include_macro=True, include_structural=True,
                          include_temporal=True,
                          include_linguistic=True, include_textual=True, include_graphical=True):
    file_names = [news_source]
    if include_micro:
        file_names.append("micro")

    if include_macro:
        file_names.append("macro")

    if include_structural:
        file_names.append("struct")

    if include_temporal:
        file_names.append("temp")

    if include_linguistic:
        file_names.append("linguistic")

    if include_textual:
        file_names.append("textual")

    if include_graphical:
        file_names.append("graphical")

    return "{}/{}".format(file_dir, "_".join(file_names))


def get_TPNF_dataset(out_dir, news_source, include_micro=True, include_macro=True, include_structural=None,
                     include_temporal=None,
                     include_linguistic=None, include_textual=None, include_graphical=None, time_interval=None,
                     use_cache=False):
    file_name = get_dataset_file_name(out_dir, news_source, include_micro, include_macro, include_structural,
                                      include_temporal, include_linguistic, include_textual, include_graphical)
    file_name = sys.path[4] + "/" + file_name
    data_file = Path(file_name + "_fake_sample_features.pkl")

    if use_cache and data_file.is_file():
        fake_sample_features = pickle.load(open(file_name + "_fake_sample_features.pkl", "rb"))
        real_sample_features = pickle.load(open(file_name + "_real_sample_features.pkl", "rb"))

        # return pickle.load(open(file_name, "rb"))

    else:
        fake_sample_features, real_sample_features = get_dataset_feature_array(news_source, include_micro,
                                                                               include_macro, include_structural,
                                                                               include_temporal, include_linguistic,
                                                                               include_textual, include_graphical,
                                                                               time_interval)

        # sample_features = np.concatenate([fake_sample_features, real_sample_features], axis=0)
        pickle.dump(fake_sample_features, open(file_name + "_fake_sample_features.pkl", "wb"))
        pickle.dump(real_sample_features, open(file_name + "_real_sample_features.pkl", "wb"))

    return fake_sample_features, real_sample_features


def get_TPNF_dataset_global(out_dir, news_source, include_micro=True, include_macro=True, include_structural=None,
                            include_temporal=None,
                            include_linguistic=None, include_textual=None, include_graphical=None, time_interval=None,
                            use_cache=False):
    file_name = get_dataset_file_name(out_dir, news_source, include_micro, include_macro, include_structural,
                                      include_temporal, include_linguistic, include_textual, include_graphical)
    file_name = sys.path[4] + "/" + file_name
    data_file = Path(file_name + "_sample_features.pkl")

    if use_cache and data_file.is_file():
        sample_features = pickle.load(open(file_name + "_sample_features.pkl", "rb"))
    else:
        sample_features = get_dataset_feature_array_global(news_source, include_micro,
                                                           include_macro, include_structural,
                                                           include_temporal, include_linguistic,
                                                           include_textual, include_graphical,
                                                           time_interval)

        # sample_features = np.concatenate([fake_sample_features, real_sample_features], axis=0)
        pickle.dump(sample_features, open(file_name + "_sample_features.pkl", "wb"))

    return sample_features


def get_dataset_feature_names(include_micro=True, include_macro=True, include_structural=None,
                              include_temporal=None,
                              include_linguistic=None, include_textual=None, include_graphical=None):
    feature_helpers = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())

    if include_linguistic:
        feature_helpers.append(LinguisticFeatureHelper())

    if include_textual:
        feature_helpers.append(TextualFeatureHelper())

    if include_graphical:
        feature_helpers.append(GraphicalFeatureHelper())

    feature_names_all = []
    short_feature_names_all = []

    for idx, feature_helper in enumerate(feature_helpers):
        features_names, short_feature_names = feature_helper.get_feature_names(include_micro, include_macro)

        feature_names_all.extend(features_names)
        short_feature_names_all.extend(short_feature_names)

    return feature_names_all, short_feature_names_all


def is_valid_graph(prop_graph: tweet_node, retweet=True, reply=True):
    """ Check if the prop graph has alteast one retweet or reply"""

    for post_node in prop_graph.children:
        if (retweet and len(post_node.reply_children) > 0) or (reply and len(post_node.retweet_children) > 0):
            return True

    return False


def remove_node_by_time(graph: tweet_node, limit_time):
    start_time = get_first_post_time(graph)
    end_time = start_time + limit_time

    q = queue.Queue()

    q.put(graph)

    while q.qsize() != 0:
        node = q.get()

        children = node.children

        retweet_children = set(node.retweet_children)
        reply_children = set(node.reply_children)

        for child in children.copy():

            if child.created_time <= end_time:
                q.put(child)
            else:
                node.children.remove(child)
                try:
                    retweet_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass
                try:
                    reply_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass

        node.retweet_children = list(retweet_children)
        node.reply_children = list(reply_children)

    return graph


def filter_propagation_graphs(graphs, limit_time):
    result_graphs = []

    for prop_graph in graphs:
        filtered_prop_graph = remove_node_by_time(prop_graph, limit_time)
        if is_valid_graph(filtered_prop_graph):
            result_graphs.append(filtered_prop_graph)

    return result_graphs


def get_nx_propagation_graphs(data_folder, news_source):
    fake_propagation_graphs = load_from_nx_graphs(data_folder, news_source, "fake")
    real_propagation_graphs = load_from_nx_graphs(data_folder, news_source, "real")

    return fake_propagation_graphs, real_propagation_graphs


def get_nx_propagation_graphs_mix(news_source):
    propagation_graphs = load_from_nx_graphs_mix(news_source)
    return propagation_graphs


def get_politifact_nx_graphs(news_source):
    fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/nx_network_data", news_source)
    all_graph = fake_prop_graph + real_prop_graph
    # pickle.dump(all_graph, open(file_name, "wb"))
    return all_graph


def get_dataset_feature_array(news_source, include_micro=True, include_macro=True, include_structural=None,
                              include_temporal=None,
                              include_linguistic=None, include_textual=None, include_graphical=None,
                              time_interval=None):
    proj_path = sys.path[4]
    fake_prop_graph_path = "/pickle/fake_prop_graph" + news_source + ".pkl"
    real_prop_graph_path = "/pickle/real_prop_graph" + news_source + ".pkl"

    if Path(proj_path + fake_prop_graph_path).is_file():
        fake_prop_graph = pickle.load(open(proj_path + fake_prop_graph_path, "rb"))
        real_prop_graph = pickle.load(open(proj_path + real_prop_graph_path, "rb"))
    else:
        fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/nx_network_data", news_source)
        pickle.dump(fake_prop_graph, open(proj_path + fake_prop_graph_path, "wb"))
        pickle.dump(real_prop_graph, open(proj_path + real_prop_graph_path, "wb"))

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    if time_interval is not None:
        time_limit = time_interval * 60 * 60

        print("Time limit in seconds : {}".format(time_limit))

        fake_prop_graph = filter_propagation_graphs(fake_prop_graph, time_limit)
        real_prop_graph = filter_propagation_graphs(real_prop_graph, time_limit)

        print("After time based filtering ")
        print("No. of fake samples : {}  No. of real samples: {}".format(len(fake_prop_graph), len(real_prop_graph)))

        fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    feature_helpers = []
    feature_group_names = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())
        feature_group_names.append("Structural")

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())
        feature_group_names.append("Temporal")

    if include_linguistic:
        feature_helpers.append(LinguisticFeatureHelper())
        feature_group_names.append("Linguistic")

    if include_textual:
        feature_helpers.append(TextualFeatureHelper())
        feature_group_names.append("Textual")

    if include_graphical:
        feature_helpers.append(GraphicalFeatureHelper())
        feature_group_names.append("Graphical")

    fake_feature_all = []
    real_feature_all = []
    for idx, feature_helper in enumerate(feature_helpers):
        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="fake")
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="real")

        feature_names = feature_helper.get_feature_names(micro_features=include_micro, macro_features=include_macro)
        print(feature_names)
        if fake_features is not None and real_features is not None:
            fake_feature_all.append(fake_features)
            real_feature_all.append(real_features)

            print("Feature group : {}".format(feature_group_names[idx]))
            print(len(fake_features))
            print(len(real_features), flush=True)

    return np.concatenate(fake_feature_all, axis=1), np.concatenate(real_feature_all, axis=1)

def compare_m(news_source, label, prop_graphs, function_refs):
    all_features = []
    last_index_visit = 0
    for function_reference in tqdm(function_refs):
        func_to_string = str(function_reference).split(" ")[1]
        file_name_pickle = sys.path[4] + "/pickle/" + func_to_string + "_" + news_source + "_" + label + ".pkl"
        if Path(file_name_pickle).is_file():
            features_set = pickle.load(open(file_name_pickle, "rb"))
        else:
            features_set = get_sample_feature_value(prop_graphs, function_reference)
            pickle.dump(features_set, open(file_name_pickle, "wb"))

        if isinstance(features_set[0], dict):
            for feature in features_set:
                if isinstance(feature, int):
                    feature_arr = [
                        {'anger': 0.0, 'anticipation': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'negative': 0.0,
                         'positive': 0.0, 'sadness': 0.0, 'surprise': 0.0, 'trust': 0.0}]
                else:
                    feature_arr = list(feature.values())
                for index, item in enumerate(feature_arr):
                    if last_index_visit + index < len(all_features):
                        all_features[last_index_visit + index].append(item)
                    else:
                        all_features.append([item])
        else:
            all_features.append(features_set)

        last_index_visit = len(all_features)

    feature_array = np.transpose(get_numpy_array(all_features))
    return feature_array


def get_dataset_feature_array_global(news_source, include_micro=True, include_macro=True, include_structural=None,
                                     include_temporal=None, include_linguistic=None, include_textual=None,
                                     include_graphical=None, time_interval=None):
    proj_path = sys.path[4]
    prop_graph_path = "/pickle/" + news_source + "_all_prop_graph.pkl"

    if Path(proj_path + prop_graph_path).is_file():
        prop_graph_val = pickle.load(open(proj_path + prop_graph_path, "rb"))
    else:
        prop_graph_val = get_nx_propagation_graphs_mix(news_source)
        pickle.dump(prop_graph_val, open(proj_path + prop_graph_path, "wb"))

    if time_interval is not None:
        time_limit = time_interval * 60 * 60

        print("Time limit in seconds : {}".format(time_limit))

        prop_graph_val = filter_propagation_graphs(prop_graph_val, time_limit)

        # print("After time based filtering ")
        # print("No. of fake samples : {}  No. of real samples: {}".format(len(fake_prop_graph), len(real_prop_graph)))

        # fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    feature_helpers = []
    feature_group_names = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())
        feature_group_names.append("Structural")

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())
        feature_group_names.append("Temporal")

    # if include_linguistic:
    #     feature_helpers.append(LinguisticFeatureHelper())
    #     feature_group_names.append("Linguistic")

    if include_textual:
        feature_helpers.append(TextualFeatureHelper())
        feature_group_names.append("Textual")

    if include_graphical:
        feature_helpers.append(GraphicalFeatureHelper())
        feature_group_names.append("Graphical")

    feature_all = []
    for idx, feature_helper in enumerate(feature_helpers):
        features_array = feature_helper.get_features_array(prop_graph_val, micro_features=include_micro,
                                                           macro_features=include_macro, news_source=news_source,
                                                           label="global", use_cache=True)

        feature_names = feature_helper.get_feature_names(micro_features=include_micro, macro_features=include_macro)
        print(feature_names)
        if features_array is not None:
            feature_all.append(features_array)

            print("Feature group : {}".format(feature_group_names[idx]))
            print(len(features_array))
            # print(len(real_features), flush=True)

    return np.concatenate(feature_all, axis=1)


def get_dataset_stats_gossipcop(news_source):
    file_name = get_dataset_file_name("data/features", news_source, True, True, True,
                                      True, True, True, True)
    file_name = sys.path[4] + "/" + file_name
    data_file = Path(file_name + "_fake_sample_features.pkl")

    fake_prop_graph = pickle.load(open(file_name + "_fake_sample_features.pkl", "rb"))
    real_prop_graph = pickle.load(open(file_name + "_real_sample_features.pkl", "rb"))

    feature_helpers = [StructureFeatureHelper(), TemporalFeatureHelper(), LinguisticFeatureHelper(),
                       TextualFeatureHelper(), GraphicalFeatureHelper()]

    feature_group_names = ["StructureFeatureHelper", "TemporalFeatureHelper", "LinguisticFeatureHelper",
                           "TextualFeatureHelper", "GraphicalFeatureHelper"]

    for idx, feature_helper in enumerate(feature_helpers):
        print("Feature group : {}".format(feature_group_names[idx]))

        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="fake", use_cache=True)
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="real", use_cache=True)

        # feature_helper.save_blox_plots_for_features(fake_feature_array=fake_features,
        #                                             real_feature_array=real_features, micro_features=True,
        #                                             macro_features=True,
        #                                             save_folder=sys.path[4] + "/data/feature_images/{}".format(
        #                                                 news_source))

        feature_helper.get_feature_significance_t_tests(fake_features, real_features, micro_features=True,
                                                        macro_features=True)

        # Print the statistics of the dataset
        print("------------Fake------------")
        feature_helper.print_statistics_for_all_features(feature_array=fake_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)

        print("------------Real------------")
        feature_helper.print_statistics_for_all_features(feature_array=real_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)

def get_dataset_statistics_data(news_source):
    # file_name = get_dataset_file_name("data/features", news_source, True, True, True,
    #                                   True, True, True, True)
    # file_name = sys.path[4] + "/" + file_name
    # data_file = Path(file_name + "_fake_sample_features.pkl")
    #
    # # fake_fea = get_features_array_forme(news_source, label="fake")
    # fake_sample_features, real_sample_features = get_TPNF_dataset("data/features", news_source, True,
    #                                                                        True, True,
    #                                                                        True, True,
    #                                                                        True, True,
    #                                                                        None)

    # sample_features = np.concatenate([fake_sample_features, real_sample_features], axis=0)
    # pickle.dump(fake_sample_features, open(file_name + "_fake_sample_features.pkl", "wb"))
    # pickle.dump(real_sample_features, open(file_name + "_real_sample_features.pkl", "wb"))


    fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/nx_network_data", news_source)
    # get_features_array()
    # fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    # feature_helpers = [
    #                    GraphicalFeatureHelper()]

    feature_helpers = [StructureFeatureHelper(), TemporalFeatureHelper(), LinguisticFeatureHelper(),
                       TextualFeatureHelper(), GraphicalFeatureHelper()]

    # feature_group_names = [
    #                        "GraphicalFeatureHelper"]

    feature_group_names = ["StructureFeatureHelper", "TemporalFeatureHelper", "LinguisticFeatureHelper",
                           "TextualFeatureHelper", "GraphicalFeatureHelper"]

    for idx, feature_helper in enumerate(feature_helpers):
        print("Feature group : {}".format(feature_group_names[idx]))

        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="fake", use_cache=True)
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="real", use_cache=True)

        feature_helper.save_blox_plots_for_features(fake_feature_array=fake_features,
                                                    real_feature_array=real_features, micro_features=True,
                                                    macro_features=True,
                                                    save_folder=sys.path[4] + "/data/feature_images/{}".format(
                                                        news_source))

        feature_helper.get_feature_significance_t_tests(fake_features, real_features, micro_features=True,
                                                        macro_features=True)

        # Print the statistics of the dataset
        print("------------Fake------------")
        feature_helper.print_statistics_for_all_features(feature_array=fake_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)

        print("------------Real------------")
        feature_helper.print_statistics_for_all_features(feature_array=real_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)


if __name__ == "__main__":
    # get_politifact_nx_graphs("politifact")
    # get_dataset_statistics("politifact")
    # get_dataset_statistics("gossipcop")
    get_dataset_statistics_data("gossipcop")
    # get_dataset("politifact")
