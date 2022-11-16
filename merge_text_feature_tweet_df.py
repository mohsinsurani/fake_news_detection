import numpy as np
import pandas as pd
import ast
import pickle
from pathlib import Path
import sys
import json

df_path = sys.path[0] + "/pickle/merged_tweet_df.pkl"


def process_dfs():
    tweets_politifact_df = pd.read_csv('tweets_politifact.csv')
    politifact_df = pd.read_csv('politifact_prop.csv')

    tweets_politifact_df = tweets_politifact_df.rename(columns={'tweet': 'tweet_org'})

    politifact_df = politifact_df.set_index('id')
    tweets_politifact_df = tweets_politifact_df.set_index('id')

    df = politifact_df.join(tweets_politifact_df, on='id', how='left')
    df["tweet_mod"] = [np.empty(0, dtype=float)] * len(df)
    df = df.drop_duplicates(subset=['text', "title"], keep=False)
    df.to_pickle(df_path)
    return df


def update_tweets(title, text, tweet, tweet_org):
    tweet_dict = ast.literal_eval(tweet)
    tweet_org_arr = ast.literal_eval(tweet_org)
    tweet_dict["title"] = title
    tweet_dict["text"] = text
    tweet_arr = tweet_dict["children"]
    for index, item in enumerate(tweet_arr):
        tweet_id = item["tweet_id"]
        tweet_dic = [tweet_dic for tweet_dic in tweet_org_arr if tweet_dic["tweet_id"] == tweet_id]
        tweet_arr[index]["tweet"] = tweet_dic[0]["text"]
    tweet_dict["tweet"] = tweet_arr
    return [tweet_dict]


def process_tweet():
    if Path(df_path).is_file():
        df = pickle.load(open(df_path, "rb"))
    else:
        df = process_dfs()

    for index, item in df.iterrows():
        tweet_update_dict = update_tweets(item["title"], item["text"], item["tweet"], item["tweet_org"])
        print(index)
        if index == "politifact14940":
            print(tweet_update_dict)
        df.at[index, 'tweet_mod'] = tweet_update_dict

    df.to_pickle(df_path)


def print_error():
    df = pickle.load(open(df_path, "rb"))
    print(df[df.index == "politifact14940"])


if __name__ == '__main__':
    process_tweet()
    # print_error()