import errno
import os
import pickle
import sys
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

from propagation.util.util import twitter_datetime_str_to_object
from stat_test import perform_t_test, get_box_plots_mod


class BaseFeatureHelper(metaclass=ABCMeta):

    @abstractmethod
    def get_feature_group_name(self):
        pass

    @abstractmethod
    def get_micro_feature_method_references(self):
        pass

    @abstractmethod
    def get_micro_feature_method_names(self):
        pass

    @abstractmethod
    def get_micro_feature_short_names(self):
        pass

    @abstractmethod
    def get_macro_feature_method_references(self):
        pass

    @abstractmethod
    def get_macro_feature_method_names(self):
        pass

    @abstractmethod
    def get_macro_feature_short_names(self):
        pass

    def get_dump_file_name(self, news_source, micro_features, macro_features, label, file_dir):
        file_tags = [news_source, label, self.get_feature_group_name()]
        if micro_features:
            file_tags.append("micro")

        if macro_features:
            file_tags.append("macro")
        data_dir = sys.path[4] + "/" + file_dir
        return "{}/{}.pkl".format(data_dir, "_".join(file_tags))

    def get_features_array(self, prop_graphs, micro_features, macro_features, news_source=None, label=None,
                           file_dir="data/features", use_cache=False):
        function_refs = []

        file_name = self.get_dump_file_name(news_source, micro_features, macro_features, label, file_dir)
        data_file = Path(file_name)

        if use_cache and data_file.is_file():
            return pickle.load(open(file_name, "rb"))

        if micro_features:
            function_refs.extend(self.get_micro_feature_method_references())

        if macro_features:
            function_refs.extend(self.get_macro_feature_method_references())

        if len(function_refs) == 0:
            return None

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

        pickle.dump(feature_array, open(file_name, "wb"))

        return feature_array

    def get_feature_names(self, micro_features, macro_features):
        features_names = []
        short_feature_names = []

        if micro_features:
            features_names.extend(self.get_micro_feature_method_names())
            short_feature_names.extend(self.get_micro_feature_short_names())

        if macro_features:
            features_names.extend(self.get_macro_feature_method_names())
            short_feature_names.extend(self.get_macro_feature_short_names())

        return features_names, short_feature_names

    def print_statistics_for_all_features(self, feature_array=None, prop_graphs=None, micro_features=None,
                                          macro_features=None):

        if feature_array is None:
            feature_array = self.get_features_array(prop_graphs, micro_features, macro_features)

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            feature_values = feature_array[:, idx]
            print_stat_values(feature_names[idx], feature_values, short_feature_names[idx])

    def save_blox_plots_for_features(self, fake_feature_array=None, real_feature_array=None, fake_prop_graphs=None,
                                     real_prop_graphs=None, micro_features=None, macro_features=None, save_folder=None):

        if fake_feature_array is None:
            fake_feature_array = self.get_features_array(fake_prop_graphs, micro_features, macro_features)
            real_feature_array = self.get_features_array(real_prop_graphs, micro_features, macro_features)

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            fake_feature_values = fake_feature_array[:, idx]
            real_feature_values = real_feature_array[:, idx]
            # short_feature_names[idx]
            get_box_plots_mod(fake_feature_values, real_feature_values, save_folder, feature_names[idx])

    def get_feature_significance_t_tests(self, fake_feature_array, real_feature_array, micro_features=None,
                                         macro_features=None):
        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            fake_feature_values = fake_feature_array[:, idx]
            real_feature_values = real_feature_array[:, idx]
            print("Feature {} : {}".format(short_feature_names[idx], feature_names[idx]))
            perform_t_test(fake_feature_values, real_feature_values)

    def get_feature_significance_bootstrap_tests(self, fake_feature_array, real_feature_array, micro_features=None,
                                                 macro_features=None):

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            fake_feature_values = fake_feature_array[:, idx]
            real_feature_values = real_feature_array[:, idx]

            perms_fake = []
            perms_real = []

            combined = np.concatenate((fake_feature_values, real_feature_values), axis=0)

            print("combined shape : ", combined.shape)

            for i in range(10000):
                np.random.seed(i)
                perms_fake.append(resample(combined, n_samples=len(fake_feature_values)))
                perms_real.append(resample(combined, n_samples=len(real_feature_values)))

            dif_bootstrap_means = (np.mean(perms_fake, axis=1) - np.mean(perms_real, axis=1))
            print("diff bootstrap means : ", dif_bootstrap_means.shape)

            obs_difs = (np.mean(fake_feature_values) - np.mean(real_feature_values))

            p_value = dif_bootstrap_means[dif_bootstrap_means >= obs_difs].shape[0] / 10000

            print("Feature {} : {}".format(short_feature_names[idx], feature_names[idx]))
            print("t- value : {}   p-value : {}".format(obs_difs, p_value))


def get_sample_feature_value(news_graps: list, get_feature_fun_ref):
    result = []
    for graph in tqdm(news_graps):
        feature_dict = get_feature_fun_ref(graph)
        result.append(feature_dict)

    return result


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_epoch_timestamp_from_retweet(retweet):
    return twitter_datetime_str_to_object(retweet["created_at"])


def sort_retweet_object_by_time(retweets: list):
    retweets.sort(key=get_epoch_timestamp_from_retweet)

    return retweets


def get_noise_news_ids():
    with open("data/news_id_ignore_list") as file:
        lines = file.readlines()
        return [line.strip() for line in lines]


def load_prop_graph(data_folder, news_source, news_label):
    news_graphs = pickle.load(open("{}/{}_{}_news_prop_graphs.pkl".format(data_folder, news_source, news_label), "rb"))
    return news_graphs


def remove_prop_graph_noise(news_graphs, noise_ids):
    noise_ids = set(noise_ids)
    return [graph for graph in news_graphs if graph.tweet_id not in noise_ids]


def sort_tweet_node_object_by_created_time(tweet_nodes: list):
    tweet_nodes.sort(key=lambda x: x.created_time)

    return tweet_nodes


def equal_samples(sample1, sample2):
    target_len = min(len(sample1), len(sample2))

    np.random.seed(0)

    np.random.shuffle(sample1)
    np.random.shuffle(sample2)

    return sample1[:target_len], sample2[:target_len]


# def get_propagation_graphs(data_folder, news_source):
#     fake_propagation_graphs = load_prop_graph(data_folder, news_source, "fake")
#     real_propagation_graphs = load_prop_graph(data_folder, news_source, "real")
#
#     print("Before filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
#     print("Before filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))
#
#     fake_propagation_graphs = remove_prop_graph_noise(fake_propagation_graphs, get_noise_news_ids())
#     real_propagation_graphs = remove_prop_graph_noise(real_propagation_graphs, get_noise_news_ids())
#
#     print("After filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
#     print("After filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))
#     print(flush=True)
#
#     return fake_propagation_graphs, real_propagation_graphs


def get_numpy_array(list_of_list):
    np_array_lists = []
    for list_obj in list_of_list:
        np_array_lists.append(np.array(list_obj))

    return np.array(np_array_lists)


def print_stat_values(feature_name, values, short_feature_name=""):
    print("=========================================")
    print("Feature {} : {}".format(short_feature_name, feature_name))
    print("Min value : {}".format(min(values)))
    print("Max value : {}".format(max(values)))
    print("Mean value : {}".format(np.mean(np.array(values))))
    print("=========================================")
