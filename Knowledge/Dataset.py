import pandas as pd
import matplotlib.pyplot as plt
import pycld2 as cld2
import re
PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’'
from nltk.corpus import stopwords
stopword = stopwords.words('english')
from nltk.stem import WordNetLemmatizer


class Dataset:
    def __init__(self):
        dataset = pd.read_csv('Knowledge/twcs.csv')
        inbound = dataset[dataset.inbound == True & pd.isnull(dataset.in_response_to_tweet_id)]
        outbounds = dataset[dataset.inbound == False]
        self.tweets = pd.merge(inbound, outbounds, left_on='tweet_id', right_on='in_response_to_tweet_id')
        self.tweets['inbound_time'] = pd.to_datetime(self.tweets['created_at_x'], format='%a %b %d %H:%M:%S +0000 %Y')
        self.tweets['outbound_time'] = pd.to_datetime(self.tweets['created_at_y'], format='%a %b %d %H:%M:%S +0000 %Y')
        self.tweets['response_time'] = self.tweets['outbound_time'] - self.tweets['inbound_time']

    def showAll(self):
        self.tweets.groupby('author_id_y').text_y.count().plot(kind='bar', figsize=(20, 10), color='red', width=0.75)
        plt.show()

    def getTweets(self, company):
        tweetsCompany = self.tweets[self.tweets.author_id_y == company][:]
        tweetsCompany['lang'] = tweetsCompany['text_x'].apply(self.etl_1)
        tweetsCompany = tweetsCompany[tweetsCompany.lang == 'en'][:]
        tweetsCompany['text_x_v2'] = tweetsCompany['text_x'].apply(self.etl_2)
        tweetsCompany['text_y_v2'] = tweetsCompany['text_y'].apply(self.etl_3)
        return tweetsCompany[['text_x_v2', 'text_y_v2']]

    def etl_1(self, text):
        details = cld2.detect(text)
        text = str(details).split(", ")[3].replace("'", "")
        return text

    def etl_2(self, text):
        text = " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()) #remove links
        text = text.lower().translate(str.maketrans('', '', PUNCT_TO_REMOVE))
        text = " ".join([word for word in str(text).split() if word not in stopword])
        text = " ".join([WordNetLemmatizer().lemmatize(word, "v") for word in text.split()])
        return text

    def etl_3(self, text):
        return re.sub("@[^\s]+", '', text) #remove @username