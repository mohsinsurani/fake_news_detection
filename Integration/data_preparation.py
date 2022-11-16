import pandas as pd


class DataPreparation:
    @property
    def integrate_dfs(self):
        politifact_df = pd.read_csv('politifact.csv')
        tweets_politifact_df = pd.read_csv('tweets_politifact.csv')
        politifact_df = politifact_df.set_index('id')
        tweets_politifact_df = tweets_politifact_df.set_index('id')
        int_tweets_df = tweets_politifact_df.join(politifact_df, on='id', how='left')
        int_tweets_df.drop_duplicates(subset=['text', "title"], keep=False)
        int_tweets_df = int_tweets_df[int_tweets_df['text'].notna()]
        int_tweets_df = int_tweets_df[int_tweets_df['top_img'].notna()]
        return int_tweets_df


if __name__ == '__main__':
    DataPreparation().integrate_dfs()
