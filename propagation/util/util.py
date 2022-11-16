import datetime
import time


class tweet_node:

    def __init__(self, tweet_id, text=None, news_title=None, news_text=None, created_time=None, user_name=None,
                 user_id=None, news_id=None, node_type=None, botometer_score=None, sentiment=None, rank=None,
                 close_centre=None, constraint=None, effective_size_score=None, edge_centre=None, strong_conn=None,
                 attr_conn=None, branch_weight=None):
        self.tweet_id = tweet_id
        self.text = text
        self.news_title = news_title
        self.news_text = news_text

        self.rank = rank
        self.close_centre = close_centre
        self.constraint = constraint
        self.edge_centre = edge_centre
        self.effective_size_score = effective_size_score
        self.strong_conn = strong_conn
        self.attr_conn = attr_conn
        self.branch_weight = branch_weight

        self.created_time = created_time
        self.user_name = user_name
        self.user_id = user_id

        self.news_id = news_id

        self.retweet_children = []
        self.reply_children = []
        self.children = set()

        self.sentiment = sentiment
        self.parent_node = None
        self.node_type = node_type
        self.botometer_score = botometer_score

    def __eq__(self, other):
        return self.tweet_id == other.tweet_id

    def __hash__(self):
        return hash(self.tweet_id)

    def set_node_type(self, node_type):
        self.node_type = node_type

    def set_parent_node(self, parent_node):
        self.parent_node = parent_node

    def add_retweet_child(self, child_node):
        self.retweet_children.append(child_node)
        self.children.add(child_node)

    def add_reply_child(self, child_node):
        self.reply_children.append(child_node)
        self.children.add(child_node)

    def get_contents(self):
        return {"tweet_id": str(self.tweet_id),
                "text": self.text,
                "created_time": self.created_time,
                "user_name": self.user_name,
                "user_id": self.user_id,
                "news_id": self.news_id
                }


def twitter_datetime_str_to_object(date_str):
    time_struct = time.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")
    date = datetime.datetime.fromtimestamp(time.mktime(time_struct))
    return int(date.timestamp())
