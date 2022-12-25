import pickle
import sys
import time
from pathlib import Path
import os

import matplotlib
import numpy as np
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd

from construct_sample_features import get_TPNF_dataset, get_train_test_split, get_dataset_feature_names, \
    get_TPNF_dataset_global, get_dataset_file_name
import matplotlib.pyplot as plt

from propagation.graphical_analysis import GraphicalFeatureHelper
from propagation.linguistic_analysis import LinguisticFeatureHelper
from propagation.structure_temp_analysis import StructureFeatureHelper
from propagation.temporal_analysis import TemporalFeatureHelper
from propagation.textual_analysis import TextualFeatureHelper

matplotlib.use('agg')


class ClassficationStats:
    arr_results = []

    # def get_classifier_by_name(self, classifier_name):
    #     if classifier_name == "GaussianNB":
    #         return GaussianNB()
    #     elif classifier_name == "LogisticRegression":
    #         return LogisticRegression(solver='lbfgs')
    #     elif classifier_name == "DecisionTreeClassifier":
    #         return DecisionTreeClassifier()
    #     elif classifier_name == "RandomForestClassifier":
    #         return RandomForestClassifier(n_estimators=50)
    #     elif classifier_name == "SVM -linear kernel":
    #         return svm.SVC(kernel='linear')

    def get_classifier_by_name(self, classifier_name):
        cv = 4
        if classifier_name == "GaussianNB":
            parameters = {
                'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14,
                                  1e-15]
            }
            clf = GridSearchCV(GaussianNB(), parameters, cv=cv, verbose=True, n_jobs=-1, scoring='roc_auc')
            return clf
        elif classifier_name == "LogisticRegression":
            param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge

            clf = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=cv, verbose=True, n_jobs=-1,
                               scoring='roc_auc')
            return clf
        elif classifier_name == "DecisionTreeClassifier":
            param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                          'ccp_alpha': [0.1, .01, .001],
                          'max_depth': [3, 2],
                          'criterion': ['gini', 'entropy']
                          }
            tree_clas = DecisionTreeClassifier(random_state=124)
            grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=cv, verbose=True,
                                       scoring='roc_auc')
            return grid_search
        elif classifier_name == "RandomForestClassifier":
            param_grid = {
                'n_estimators': [50, 200, 500],
                'max_features': ['auto', 'sqrt', 'log2'],
                'ccp_alpha': [0.1, .01, .001],
                'max_depth': [4, 5, 6, 7],
                'criterion': ['gini', 'entropy']
            }
            rfc_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=443), param_grid=param_grid, cv=cv,
                                    scoring='roc_auc')
            return rfc_grid
        elif classifier_name == "SVM -linear kernel":
            param_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                           'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                          {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                           'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                          {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1]}
                          ]
            grid = GridSearchCV(svm.SVC(probability=True), param_grid, refit=True, cv=cv, verbose=3, scoring='roc_auc')
            return grid
        elif classifier_name == "XGBClassifier":
            gs_xgb = XGBClassifier(
                eta=0.3,
                n_estimators=500,
                gamma=[0, 0.5, 1],
                max_depth=4,
                min_child_weight=1,
                colsample_bytree=1,
                colsample_bylevel=1,
                subsample=1,
                seed=233
            )

            param_grid = [
                {'booster': ['gbtree', 'dart'], 'n_estimators': [100, 150, 200, 250], 'max_depth': [5, 6, 7]}
            ]
            grid = GridSearchCV(gs_xgb, param_grid, refit=True, cv=cv, verbose=3, scoring='roc_auc')
            return grid
        elif classifier_name == "LGBMClassifier":
            params = {
                'application': 'binary',  # for binary classification
                'boosting': 'gbdt',  # traditional gradient boosting decision tree
                'num_iterations': 10,
                'learning_rate': 0.05,
                'num_leaves': 62,
                'max_depth': 4,  # <0 means no limit
                'max_bin': 510,  # Small number of bins may reduce training accuracy but can deal with over-fitting
                'lambda_l1': 5,  # L1 regularization
                'lambda_l2': 10,  # L2 regularization
                'metric': 'binary_error',
                'subsample_for_bin': 200,  # number of samples for constructing bins
                'subsample': 1,  # subsample ratio of the training instance
                'colsample_bytree': 0.8,  # subsample ratio of columns when constructing the tree
                'min_split_gain': 0.5,
                # minimum loss reduction required to make further partition on a leaf node of the tree
                'min_child_weight': 1,  # minimum sum of instance weight (hessian) needed in a leaf
                'min_child_samples': 5  # minimum number of data needed in a leaf
            }

            gs_lgb = LGBMClassifier(boosting_type='gbdt',
                                    objective='binary',
                                    n_jobs=5,
                                    silent=True,
                                    max_depth=params['max_depth'],
                                    max_bin=params['max_bin'],
                                    subsample_for_bin=params['subsample_for_bin'],
                                    subsample=params['subsample'],
                                    min_split_gain=params['min_split_gain'],
                                    min_child_weight=params['min_child_weight'],
                                    min_child_samples=params['min_child_samples'])

            # scale of tree
            param_grid = {
                'learning_rate': [0.005, 0.01],
                'n_estimators': [8, 16, 24],
                'num_leaves': [6, 8, 12, 16],  # large num_leaves helps improve accuracy but might lead to over-fitting
                'boosting_type': ['gbdt', 'dart'],  # for better accuracy -> try dart
                'objective': ['binary'],
                'max_bin': [255, 510],  # large max_bin helps improve accuracy but might slow down training progress
                'random_state': [500],
                'colsample_bytree': [0.64, 0.65, 0.66],
                'subsample': [0.7, 0.75],
                'reg_alpha': [1, 1.2],
                'reg_lambda': [1, 1.2, 1.4],
            }

            grid = GridSearchCV(gs_lgb, param_grid, refit=True, cv=cv, verbose=3, scoring='roc_auc')
            return grid

    # def train_model(self, classifier_name, X_train, X_test, y_train, y_test):
    #     accuracy_values = []
    #     precision_values = []
    #     recall_values = []
    #     f1_score_values = []
    #
    #     for i in range(5):
    #         classifier_clone = self.get_classifier_by_name(classifier_name)
    #         classifier_clone.fit(X_train, y_train)
    #
    #         predicted_output = classifier_clone.predict(X_test)
    #         accuracy, precision, recall, f1_score_val = self.get_metrics(y_test, predicted_output, one_hot_rep=False)
    #
    #         accuracy_values.append(accuracy)
    #         precision_values.append(precision)
    #         recall_values.append(recall)
    #         f1_score_values.append(f1_score_val)
    #
    #     self.print_metrics(np.mean(accuracy_values), np.mean(precision_values), np.mean(recall_values),
    #                   np.mean(f1_score_values))

    def train_model(self, classifier_name, news_source, x_train, x_test, y_train, y_test, include_structural=True,
                    include_temporal=True, include_linguistic=True, include_textual=True,
                    include_graphical=True):

        pickle_path = sys.path[4] + "/pickle/"
        name = self.get_pickle_name(news_source, True,
                                    True, True, True,
                                    True)
        acutal_name = self.get_pickle_name(news_source, include_structural,
                                           include_temporal, include_linguistic, include_textual,
                                           include_graphical)
        pickle_path_classifier = pickle_path + name

        arr_source = []
        if Path(pickle_path_classifier).is_file():
            arr_source = pickle.load(open(pickle_path_classifier, "rb"))

        search_res = list(
            filter(lambda item: item['classifier'] == classifier_name and item["name"] == acutal_name, arr_source))

        if len(search_res) == 0:
            classifier_clone = self.get_classifier_by_name(classifier_name)
            classifier_clone.fit(x_train, y_train)
            predicted_output = classifier_clone.predict(x_test)
            accuracy, precision, recall, f1_score_val, roc_auc = self.get_metrics(y_test, predicted_output,
                                                                                  one_hot_rep=False)

            dict_res = {"classifier": classifier_name, "model": classifier_clone, "accuracy": accuracy,
                        "precision": precision, "recall": recall, "f1_score": f1_score_val, "roc_auc": roc_auc,
                        "news_source": news_source, "best_param": classifier_clone.best_params_,
                        "best_score": classifier_clone.best_score_, "best_estimator": classifier_clone.best_estimator_,
                        "best_index": classifier_clone.best_index_, "pred": predicted_output, "name": acutal_name}

            arr_source.append(dict_res)
            pickle.dump(arr_source, open(pickle_path_classifier, "wb"))

        print(search_res)
        # self.print_metrics(search_dict["accuracy"], search_dict["precision"], search_dict["recall"], search_dict["f1_score_val"])

    def get_pickle_name(self, news_source, include_structural=True,
                        include_temporal=True, include_linguistic=True, include_textual=True,
                        include_graphical=True):
        under_score = "_"
        name = "classifiers" + news_source
        if include_structural:
            name += under_score + "structural"
        if include_temporal:
            name += under_score + "temporal"
        if include_linguistic:
            name += under_score + "linguistic"
        if include_textual:
            name += under_score + "textual"
        if include_graphical:
            name += under_score + "graphical"
        name += ".pkl"
        return name

    def print_metrics(self, accuracy, precision, recall, f1_score_val):
        print("Accuracy : {}".format(accuracy))
        print("Precision : {}".format(precision))
        print("Recall : {}".format(recall))
        print("F1 : {}".format(f1_score_val))

    def get_metrics(self, target, logits, one_hot_rep=True):
        """
        Two numpy one hot arrays
        :param target:
        :param logits:
        :return:
        """

        if one_hot_rep:
            label = np.argmax(target, axis=1)
            predict = np.argmax(logits, axis=1)
        else:
            label = target
            predict = logits

        accuracy = accuracy_score(label, predict)

        precision = precision_score(label, predict)
        recall = recall_score(label, predict)
        f1_score_val = f1_score(label, predict)
        roc_auc = roc_auc_score(label, predict)

        return accuracy, precision, recall, f1_score_val, roc_auc

    def get_basic_model_results(self, news_source, x_train, x_test, y_train, y_test, include_structural=True,
                                include_temporal=True, include_linguistic=True, include_textual=True,
                                include_graphical=True):
        scaler = preprocessing.StandardScaler().fit(x_train)

        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model_dicts = {"GaussianNB": GaussianNB(), "LogisticRegression": LogisticRegression(),
                       "DecisionTreeClassifier": DecisionTreeClassifier(),
                       "RandomForestClassifier": RandomForestClassifier(), "SVM -linear kernel": svm.SVC(),
                       "LGBMClassifier": LGBMClassifier(), "XGBClassifier": XGBClassifier()}

        # classifiers = [GaussianNB(), LogisticRegression(), DecisionTreeClassifier(),
        #                RandomForestClassifier(n_estimators=100),
        #                svm.SVC()]
        # classifier_names = ["GaussianNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
        #                     "SVM -linear kernel"]

        for model in model_dicts.keys():
            print("======={}=======".format(model))
            self.train_model(model, news_source, x_train, x_test, y_train, y_test, include_structural,
                             include_temporal, include_linguistic, include_textual,
                             include_graphical)

        # for idx in range(len(model_dicts.keys())):
        #     print("======={}=======".format(classifier_names[idx]))
        #     self.train_model(classifier_names[idx], X_train, X_test, y_train, y_test)

    def get_classificaton_results_tpnf(self, data_dir, news_source, time_interval, use_cache=False):
        include_micro = True
        include_macro = True

        # include_structural = True
        # include_temporal = True
        # include_linguistic = True

        include_structural = True
        include_temporal = True
        include_linguistic = True
        include_textual = True
        include_graphical = True

        fake_sample_features, real_sample_features = get_TPNF_dataset(data_dir, news_source, include_micro,
                                                                      include_macro, include_structural,
                                                                      include_temporal, include_linguistic,
                                                                      include_textual,
                                                                      include_graphical,
                                                                      time_interval, use_cache=use_cache)

        print("Sample feature array dimensions")
        sample_feature_array = np.concatenate([fake_sample_features, real_sample_features], axis=0)
        print(sample_feature_array.shape, flush=True)

        target_labels = np.concatenate([np.zeros(len(fake_sample_features)),
                                        np.ones(len(real_sample_features))], axis=0)

        X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
        self.get_basic_model_results(news_source, X_train, X_test, y_train, y_test)

    def add_feature_vectors_to_df(self, data_dir, news_source):
        include_micro = True
        include_macro = True

        include_structural = True
        include_temporal = True
        include_linguistic = True
        include_textual = True
        include_graphical = True

        file_name = get_dataset_file_name(data_dir, news_source, include_micro, include_macro, include_structural,
                                          include_temporal, include_linguistic, include_textual, include_graphical)
        file_name = sys.path[4] + "/" + file_name
        sample_feature_array = pickle.load(open(file_name + "_sample_features.pkl", "rb"))
        dataset_dir = sys.path[4] + "/csvs/" + news_source + "_prop.pkl"
        struct_temp = StructureFeatureHelper()
        struct_f1 = struct_temp.get_micro_feature_short_names()
        struct_f2 = struct_temp.get_macro_feature_short_names()
        temp_features = TemporalFeatureHelper()
        temp_f1 = temp_features.get_micro_feature_short_names()
        temp_f2 = temp_features.get_macro_feature_short_names()
        # ling_features = LinguisticFeatureHelper()
        # ling_f1 = ling_features.get_micro_feature_short_names()
        text_features = TextualFeatureHelper()
        text_f1 = text_features.get_macro_feature_short_names()
        graph_features = GraphicalFeatureHelper()
        graph_f1 = graph_features.get_macro_feature_short_names()
        all_features = struct_f1 + struct_f2 + temp_f1 + temp_f2 + text_f1 + graph_f1
        print(all_features)

        df = pickle.load(open(dataset_dir, "rb"))
        fea_dict = []
        # for idx in range(0, sample_feature_array.shape[1]):
        #     fea_dict[all_features[idx]] = sample_feature_array[idx]

        print(fea_dict)
        for (x, y), value in np.ndenumerate(sample_feature_array):
            if x < len(fea_dict):
                fea_dict[x][all_features[y]] = value
            else:
                new_fea = {}
                new_fea[all_features[y]] = value
                fea_dict.append(new_fea)

        # print(fea_dict)
        # df_feature = pd.DataFrame(fea_dict)

        i = 0
        for index, item in df.iterrows():
            fea_doc = fea_dict[i]
            for key, value in fea_doc.items():
                df.at[index, key] = value
            i += 1
        print(df.head())
        df_file_name = sys.path[4] + "/csvs/" + news_source + "_global_feature.pkl"
        pickle.dump(df, open(df_file_name, "wb"))

    def get_classificaton_results_tpnf_global(self, data_dir, news_source, time_interval, use_cache=False):
        include_micro = True
        include_macro = True

        include_structural = True
        include_temporal = True
        include_linguistic = True
        include_textual = True
        include_graphical = True

        file_name = get_dataset_file_name(data_dir, news_source, include_micro, include_macro, include_structural,
                                          include_temporal, include_linguistic, include_textual, include_graphical)
        file_name = sys.path[4] + "/" + file_name
        data_file = Path(file_name + "_sample_features.pkl")

        if use_cache and data_file.is_file():
            sample_feature_array = pickle.load(open(file_name + "_sample_features.pkl", "rb"))
        else:
            sample_feature_array = get_TPNF_dataset_global(data_dir, news_source, include_micro,
                                                           include_macro, include_structural,
                                                           include_temporal, include_linguistic,
                                                           include_textual,
                                                           include_graphical,
                                                           time_interval, use_cache=use_cache)

        print("Sample feature array dimensions")
        dataset_dir = sys.path[4] + "/csvs/" + news_source + "_prop.pkl"
        df = pickle.load(open(dataset_dir, "rb"))
        target_labels = np.array(df["target"].tolist())
        # target_labels = np.concatenate([np.zeros(len(fake_sample_features)),
        #                                 np.ones(len(real_sample_features))], axis=0)

        X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
        self.get_basic_model_results(news_source, X_train, X_test, y_train, y_test)

    def get_classificaton_results_tpnf_global_featurewise(self, news_source,
                                                          include_structural=True,
                                                          include_temporal=True,
                                                          include_linguistic=True,
                                                          include_textual=True,
                                                          include_graphical=True):

        dataset_dir = sys.path[4] + "/csvs/" + news_source + "_global_feature.pkl"
        df = pickle.load(open(dataset_dir, "rb"))

        sample_feature_array = []
        start = 0
        end = 0
        if include_structural:
            if start == 0:
                start = 10
            end = 10 + 14
        if include_temporal:
            if start == 0:
                start = 10 + 14
            end = 10 + 14 + 13
        if include_linguistic:
            if start == 0:
                start = 10 + 14 + 13
            end = 10 + 14 + 13 + 6
        if include_textual:
            if start == 0:
                start = 10 + 14 + 13 + 6
            end = 10 + 14 + 13 + 6 + 19
        if include_graphical:
            if start == 0:
                start = 10 + 14 + 13 + 6 + 19
            end = 10 + 14 + 13 + 6 + 19 + 6

        sample_feature_array = df.iloc[:, start:end]

        target_labels = np.array(df["target"].tolist())

        X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
        self.get_basic_model_results(news_source, X_train, X_test, y_train, y_test,
                                     include_structural, include_temporal, include_linguistic,
                                     include_textual, include_graphical)

    @staticmethod
    def plot_feature_importances(coef, names):
        imp = coef
        imp, names = zip(*sorted(zip(imp, names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)

        plt.savefig('feature_importance.png', bbox_inches='tight')
        plt.show()

    def dump_random_forest_feature_importance(self, data_dir, news_source):
        include_micro = True
        include_macro = True

        include_structural = True
        include_temporal = True
        include_linguistic = True
        include_textual = True
        include_graphical = True

        # sample_feature_array = get_TPNF_dataset_global(data_dir, news_source, include_micro, include_macro, include_structural,
        #                                         include_temporal, include_linguistic, use_cache=True)

        file_name = get_dataset_file_name(data_dir, news_source, include_micro, include_macro, include_structural,
                                          include_temporal, include_linguistic, include_textual, include_graphical)
        file_name = sys.path[4] + "/" + file_name
        sample_feature_array = pickle.load(open(file_name + "_sample_features.pkl", "rb"))

        sample_feature_array = sample_feature_array[:, :-1]
        feature_names, short_feature_names = get_dataset_feature_names(include_micro, include_macro, include_structural,
                                                                       include_temporal, include_linguistic,
                                                                       include_textual, include_graphical)

        feature_names = feature_names[:-1]
        short_feature_names = short_feature_names[:-1]
        num_samples = int(len(sample_feature_array) / 2)
        # target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)
        dataset_dir = sys.path[4] + "/csvs/" + news_source + "_prop.pkl"
        df = pickle.load(open(dataset_dir, "rb"))
        target_labels = np.array(df["target"].tolist())

        X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)

        # Build a forest and compute the feature importances
        # forest = ExtraTreesClassifier(n_estimators=100, random_state=0)
        forest = XGBClassifier()
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        matplotlib.rcParams['figure.figsize'] = 5, 2

        # Plot the feature importances of the forest
        plt.figure()

        plt.bar(range(X_train.shape[1]), importances[indices],
                color="b", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), np.array(short_feature_names)[indices], rotation=75, fontsize=9.5)
        plt.xlim([-1, X_train.shape[1]])
        plt.savefig('{}_feature_importance.png'.format(news_source), bbox_inches='tight')

        plt.show()

    def get_classificaton_results_tpnf_by_time(self, news_source: str):
        # Time Interval in hours for early-fake news detection
        time_intervals = [3, 6, 12, 24, 36, 48, 60, 72, 84, 96]

        for time_interval in time_intervals:
            print("=============Time Interval : {}  ==========".format(time_interval))
            start_time = time.time()
            self.get_classificaton_results_tpnf("data/features", news_source, time_interval)

            print("\n\n================Exectuion time - {} ==================================\n".format(
                time.time() - start_time))


if __name__ == "__main__":
    # ClassficationStats().get_classificaton_results_tpnf_global("data/features", "politifact", time_interval=None,
    #                                                            use_cache=False)
    # ClassficationStats().get_classificaton_results_tpnf_global("data/features", "gossipcop", time_interval=None,
    #                                                            use_cache=True)
    # ClassficationStats().get_classificaton_results_tpnf_global_featurewise("gossipcop",
    #                                                                        include_structural=False,
    #                                                                        include_temporal=False,
    #                                                                        include_linguistic=False,
    #                                                                        include_textual=True,
    #                                                                        include_graphical=True)
    # get_classificaton_results_tpnf("data/features", "gossipcop", time_interval=None, use_cache=False)

    # Filter the graphs by time interval (for early fake news detection) and get the classification results
    # get_classificaton_results_tpnf_by_time("politifact")
    # get_classificaton_results_tpnf_by_time("gossipcop")
    # ClassficationStats().add_feature_vectors_to_df("data/features", "politifact", time_interval=None,use_cache=False)
    ClassficationStats().add_feature_vectors_to_df("data/features", "gossipcop")

    # ClassficationStats().dump_random_forest_feature_importance("data/features", "politifact")
