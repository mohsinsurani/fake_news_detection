import sys
import os
import json
import pandas as pd
import numpy as np


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def process_data(path_dir, news_dirs):
    news_json_arr = []
    tweets_json_arr = []
    retweets_json_arr = []
    folder_dir = path_dir + news_dirs

    for dir_folder in os.listdir(folder_dir):
        target = 1
        if dir_folder == "fake":
            target = 0
        folder_dir_path = folder_dir + "/" + dir_folder
        for name in os.listdir(folder_dir_path):
            if name != ".DS_Store":
                file_path = folder_dir_path + "/" + name
                news_data, tweets_data, retweets_data = get_relevant_data(name, file_path, target, news_dirs)
                if news_data:
                    news_json_arr.append(news_data)
                if tweets_data:
                    tweets_json_arr.append(tweets_data)
                if retweets_data:
                    tweets_json_arr.append(retweets_data)

    return news_json_arr, tweets_json_arr, retweets_json_arr


def get_relevant_data(news_id, first_dir, target_var, news_source):

    tweets = {}
    retweets = {}
    news = {}
    news_path = first_dir + "/news content.json"
    if os.path.exists(news_path):
        with open(news_path) as json_file:
            news_json = json.load(json_file)
            if not news_json:
                return {}, {}, {}
            news["id"] = news_id
            news["text"] = news_json["text"]
            news["title"] = news_json["title"]
            news["top_img"] = news_json["top_img"]
            news["publish_date"] = news_json["publish_date"]
            news["images"] = news_json["images"]
            news["source"] = news_source
            news["target"] = target_var

    # with open(first_dir + "/tweets.json") as tweets_file:
    #     tweets_json = json.load(tweets_file)
    #     if len(tweets_json["tweets"]) > 0:
    #         tweets["id"] = news_id
    #         tweets["tweet"] = tweets_json["tweets"]
    #
    # with open(first_dir + "/retweets.json") as retweets_file:
    #     retweets_json = json.load(retweets_file)
    #     if retweets_json:
    #         retweets["id"] = news_id
    #         retweets["retweet"] = retweets_json

    return news, tweets, retweets


def process_politifact():
    dataset_directory = sys.path[0] + "/git/FakeNewsNet_Dataset/"
    news_dirs = get_immediate_subdirectories(dataset_directory)
    print(news_dirs)
    politifact_data = list(filter(lambda k: 'politifact' in k, news_dirs))
    gossipicop_data = list(filter(lambda k: 'gossipcop' in k, news_dirs))
    news_json_arr, tweets_arr, retweets_arr = process_data(dataset_directory, politifact_data[0])

    df = pd.DataFrame(news_json_arr, columns=news_json_arr[0].keys())
    df.to_csv("politifact_git.csv", encoding='utf-8', index=False)

    # tweets_df = pd.DataFrame(tweets_arr, columns=tweets_arr[0].keys())
    # tweets_df.to_csv("politifact_git_tweets_dataset.csv", encoding='utf-8', index=False)
    #
    # retweets_df = pd.DataFrame(retweets_arr, columns=retweets_arr[0].keys())
    # retweets_df.to_csv("politifact_git_retweets_dataset.csv", encoding='utf-8', index=False)

    # print(news_json_arr)



if __name__ == '__main__':
    process_politifact()


