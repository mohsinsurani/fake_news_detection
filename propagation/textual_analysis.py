import queue

import nltk
import numpy as np
from analysis_util import get_numpy_array, BaseFeatureHelper
from propagation.util.util import tweet_node
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from LeXmo import LeXmo
from urlextract import URLExtract

nltk.download('punkt')

sent_analyzer: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
extractor = URLExtract()


def get_length_of_tweet(prop_graph: tweet_node):
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


def get_num_of_url(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    num_url_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                urls = extractor.find_urls(tweet)
                if len(urls) > 0:
                    num_url_arr.append(len(urls))

    if len(num_url_arr) == 0:
        return 0
    else:
        return np.mean(np.array(num_url_arr))


def get_num_of_hashtags(prop_graph: tweet_node):
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


def get_emotion_of_tweet(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None and len(tweet) > 0:
                try:
                    emotion_dict = LeXmo.LeXmo(tweet)
                    emotion_dict.pop('text', None)
                except:
                    print(tweet)
                    emotion_dict = {'anger': 0.0, 'anticipation': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'negative': 0.0, 'positive': 0.0, 'sadness': 0.0, 'surprise': 0.0, 'trust': 0.0}
                sent_arr.append(emotion_dict)
            else:
                print(tweet)

    if len(sent_arr) == 0:
        return [{'anger': 0.0, 'anticipation': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'negative': 0.0, 'positive': 0.0, 'sadness': 0.0, 'surprise': 0.0, 'trust': 0.0}]
    else:
        return get_average_from_dict_arrays(sent_arr)


def get_sentiment_of_tweet(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    sent_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                try:
                    sent_dict = sent_analyzer.polarity_scores(tweet)
                except:
                    sent_dict = {"neg": 0.0, "pos": 0.0, "neu": 0.0, "compound": 0.0}

                sent_arr.append(sent_dict)

    if len(sent_arr) == 0:
        return 0
    else:
        return get_average_from_dict_arrays(sent_arr)


def get_average_from_dict_arrays(arr):
    emo_attr_dict = dict.fromkeys(arr[0].keys(), [])

    for sent in arr:
        for key, value in sent.items():
            arr_val = emo_attr_dict[key].copy()
            arr_val.append(value)
            emo_attr_dict[key] = arr_val

    averages = [(k, sum(v) / len(v)) for k, v in emo_attr_dict.items()]
    emo_mean_dict = {}
    for k, v in averages:
        emo_mean_dict[k] = v
    return emo_mean_dict


def get_closeness_tweet_news(prop_graph: tweet_node):
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
                    try:
                        cosine_score = get_cosine_two_sentence(tweet, text_to_cons)
                        sent_arr.append(cosine_score)
                    except:
                        print("tweet, text_to_cons")

    if len(sent_arr) == 0:
        return 0
    else:
        return np.mean(np.array(sent_arr))


def get_cosine_two_sentence(x: str, y: str):
    X_list = word_tokenize(x)
    Y_list = word_tokenize(y)

    sw = stopwords.words('english')
    l1 = []
    l2 = []

    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}
    rector = X_set.union(Y_set)
    for w in rector:
        if w in X_set:
            l1.append(1)
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    for i in range(len(rector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
    return cosine


def get_num_of_person_mentions(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    num_persons_arr = []

    while q.qsize() != 0:
        node = q.get()

        for child in node.children:
            q.put(child)
            tweet = child.text
            if tweet is not None:
                persons = {tag.strip("@") for tag in tweet.split() if tag.startswith("@")}
                if len(persons) > 0:
                    num_persons_arr.append(len(persons))

    if len(num_persons_arr) == 0:
        return 0
    else:
        return np.mean(np.array(num_persons_arr))


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
    macro_features_functions = [get_emotion_of_tweet,
                                get_length_of_tweet,
                                get_num_of_url,
                                get_closeness_tweet_news,
                                get_num_of_hashtags,
                                get_sentiment_of_tweet,
                                get_num_of_person_mentions
                                ]

    function_refs = []

    if macro_features:
        function_refs.extend(macro_features_functions)

    all_features = []

    for function_reference in function_refs:
        features_set = get_stats_for_features(prop_graphs, function_reference, print=False, feature_name=None)
        all_features.append(features_set)

    return np.transpose(get_numpy_array(all_features))


class TextualFeatureHelper(BaseFeatureHelper):

    def get_micro_feature_method_references(self):
        return []

    def get_micro_feature_method_names(self):
        return []

    def get_micro_feature_short_names(self):
        return []

    def get_feature_group_name(self):
        return "tex"

    def get_macro_feature_method_references(self):
        method_refs = [get_emotion_of_tweet,
                       get_length_of_tweet,
                       get_num_of_url,
                       get_closeness_tweet_news,
                       get_num_of_hashtags,
                       get_sentiment_of_tweet,
                       get_num_of_person_mentions
                       ]

        return method_refs

    def get_macro_feature_method_names(self):
        feature_names = ["Anger Emotion behind the tweet",
                         "anticipation Emotion behind the tweet",
                         "disgust Emotion behind the tweet",
                         "fear Emotion behind the tweet",
                         "joy Emotion behind the tweet",
                         "negative Emotion behind the tweet",
                         "positive Emotion behind the tweet",
                         "sadness Emotion behind the tweet",
                         "surprise Emotion behind the tweet",
                         "trust Emotion behind the tweet",
                         "Length of the tweet",
                         "Count of the URL",
                         "Closeness of the tweet with the news title in macro network",
                         "Number of hash tags in tweet in macro network",
                         "Neg Sentiment score of the tweet in macro network",
                         "Neu Sentiment score of the tweet in macro network",
                         "Pos Sentiment score of the tweet in macro network",
                         "Compound Sentiment score of the tweet in macro network",
                         "Number of people mentioned in macro network"
                         ]

        return feature_names

    def get_macro_feature_short_names(self):
        feature_names = []
        for fea in range(len(self.get_macro_feature_method_names())):
            feature_names.append("Sen" + str(fea + 1))
        return feature_names
