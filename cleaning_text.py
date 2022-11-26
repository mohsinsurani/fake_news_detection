import sys

import pickle
import string
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class CleaningText:
    all_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def convert_lower(self, text):
        return str(text).lower()

    def remove_stopwrds(self, text):
        return ' '.join([word for word in str(text).split() if word not in self.all_stopwords])

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def clean_txt(self, text):
        text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
        text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
        text = re.sub(r"http?://[A-Za-z0-9./]+", ' ', text)
        text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
        text = re.sub('\t', ' ', text)
        text = re.sub(r" +", ' ', text)
        return text

    def lemmatize_words(self, text):
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join(
            [self.lemmatizer.lemmatize(word, self.wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in
             pos_tagged_text])

    def perform_clean(self, text):
        basic_clean_txt = self.clean_txt(text)
        clean_txt = self.convert_lower(basic_clean_txt)
        clean_txt = self.remove_punctuation(clean_txt)
        lem_txt = self.lemmatize_words(clean_txt)
        return basic_clean_txt, lem_txt

# if __name__ == "__main__":
#     CleaningText().perform_clean()
