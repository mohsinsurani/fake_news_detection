import sys
import os
import json
import pandas as pd


class Crawling_han:

    @staticmethod
    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def process_data(self, path_dir, news_dirs):
        tweets_json_arr = []

        for dir_folder in news_dirs:
            folder_dir = path_dir + dir_folder
            for name in os.listdir(folder_dir):
                if name != ".DS_Store":
                    file_path = folder_dir + "/" + name
                    tweets_data = self.get_relevant_data(file_path)
                    if tweets_data:
                        tweets_json_arr.append(tweets_data)

        return tweets_json_arr

    def get_relevant_data(self, first_dir):
        tweets = {}

        with open(first_dir) as tweets_file:
            tweets_json = json.load(tweets_file)
            tweets["id"] = tweets_json["tweet_id"]
            tweets["tweet"] = tweets_json
            return tweets

    def process_politifact(self):
        tweets_arr = self.get_data_list('politifact')
        self.save_to_csv(tweets_arr, "politifact_han")

    def process_gossipicop(self):
        tweets_arr = self.get_data_list('gossipcop')
        self.save_to_csv(tweets_arr, "gossipicop_han")

    def get_data_list(self, data_cat):
        dataset_directory = sys.path[0] + "/data/nx_network_data/"
        news_dirs = self.get_immediate_subdirectories(dataset_directory)
        print(news_dirs)

        data = list(filter(lambda k: data_cat in k, news_dirs))
        tweets_arr = self.process_data(dataset_directory, data)
        return tweets_arr

    @staticmethod
    def save_to_csv(tweets_arr, tweets_cat):
        tweets_df = pd.DataFrame(tweets_arr, columns=tweets_arr[0].keys())
        tweets_df.to_csv(tweets_cat + ".csv", encoding='utf-8', index=False)


if __name__ == '__main__':
    Crawling_han().process_politifact()
