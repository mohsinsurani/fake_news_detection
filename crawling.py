import sys
import os
import json
import pandas as pd
import numpy as np

# path_dir = sys.path[0] + "/data/FakeNewsNet_Dataset/politifact_fake"


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def process_data(path_dir, news_dirs):
    news_json_arr = []
    tweets_json_arr = []
    retweets_json_arr = []

    for dir_folder in news_dirs:
        target = 1
        if "_fake" in dir_folder:
            target = 0
        folder_dir = path_dir + dir_folder
        for name in os.listdir(folder_dir):
            if name == ".DS_Store":
                break
            for file in os.listdir(folder_dir + "/" + name):
                if file not in [".DS_Store", "likes.json", "replies.json"]:
                    file_path = folder_dir + "/" + name
                    news_data = get_relevant_data(name, file_path, target)
                    news_json_arr.append(news_data)

    return news_json_arr, tweets_json_arr, retweets_json_arr


def get_relevant_data(news_id, first_dir, target_var):

    tweets = {}
    retweets = {}
    with open(first_dir + "/news_article.json") as json_file:
        news_json = json.load(json_file)
        if not news_json:
            return {}
        news = {"id": news_id}
        news["text"] = news_json["text"]
        news["title"] = news_json["title"]
        news["top_img"] = news_json["top_img"]
        news["publish_date"] = news_json["publish_date"]
        news["images"] = news_json["images"]
        news["source"] = news_json["news_source"]
        news["target"] = target_var

    with open(first_dir + "/tweets.json") as tweets_file:
        tweets_json = json.load(tweets_file)
        tweets["id"] = news_id
        tweets["tweet"] = tweets_json["tweets"]

    with open(first_dir + "/retweets.json") as retweets_file:
        retweets_json = json.load(retweets_file)
        retweets["id"] = news_id
        retweets["retweet"] = retweets_json
        return news, tweets, retweets


def process_politifact():
    dataset_directory = sys.path[0] + "/data/FakeNewsNet_Dataset/"
    news_dirs = get_immediate_subdirectories(dataset_directory)
    news_json_arr, tweets_arr, retweets_arr = process_data(dataset_directory, news_dirs)
    np.save('news_json_arr.npy', news_json_arr)  # save
    np.save('tweets_arr.npy', tweets_arr)  # save
    np.save('retweets_arr.npy', retweets_arr)  # save

    df = pd.DataFrame(news_json_arr, columns=news_json_arr[0].keys())
    df.to_csv("news_dataset.csv", encoding='utf-8', index=False)

    tweets_df = pd.DataFrame(tweets_arr, columns=tweets_arr[0].keys())
    tweets_df.to_csv("tweets_dataset.csv", encoding='utf-8', index=False)

    retweets_df = pd.DataFrame(retweets_arr, columns=retweets_arr[0].keys())
    retweets_df.to_csv("retweets_dataset.csv", encoding='utf-8', index=False)

    print(news_json_arr)


if __name__ == '__main__':
    process_politifact()
