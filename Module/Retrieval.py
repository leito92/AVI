from rank_bm25 import BM25Okapi
import re
PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’'
from nltk.corpus import stopwords
stopword = stopwords.words('english')
from nltk.stem import WordNetLemmatizer


class Retrieval:
    def __init__(self, dataset):
        self.db = dataset
        self.corpus = self.db['text_x_v2'].tolist()
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def getRC(self, query):
        tokenized_query = self.etl(query)
        index = self.bm25.get_top_n(tokenized_query, self.corpus, n=1)
        result = self.db[self.db.text_x_v2.isin(index)][:]
        return result['text_y_v2'].tolist()

    def etl(self, text):
        text = " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
        text = text.lower().translate(str.maketrans('', '', PUNCT_TO_REMOVE))
        text = " ".join([word for word in str(text).split() if word not in stopword])
        text = " ".join([WordNetLemmatizer().lemmatize(word, "v") for word in text.split()])
        return text.split(" ")