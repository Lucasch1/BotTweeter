# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:50:29 2022

@author: hl0001
"""

import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

log = pd.read_csv("C:/Users/lucas/Desktop/BotTweeter/log.csv")

print(log)
consumerkey = log["key"][0]
consumersecret = log["key"][1]
accestoken = log["key"][2]
accesstokensecret = log["key"][3]

authenticate = tweepy.OAuthHandler(consumerkey, consumersecret)

authenticate.set_access_token(accestoken, accesstokensecret)

api = tweepy.API(authenticate, wait_on_rate_limit=True)


search_term = "#bitcoin -filter:retweets"

tweets = tweepy.Cursor(
    api.search_tweets,
    q=search_term,
    lang="en",
    tweet_mode="extended",
).items(100)

all_tweets = [tweet.full_text for tweet in tweets]

df = pd.DataFrame(all_tweets, columns=["Tweets"])

df.head(5)


def cleantwt(twt):
    twt = re.sub("#bitcoin", "bitcoin", twt)
    twt = re.sub("#Bitcoin", "Bitcoin", twt)
    twt = re.sub("#[A-Za-z0-9]+", "", twt)
    twt = re.sub("\\n", "", twt)
    twt = re.sub("hhtps?:\/\/\S+", "", twt)

    return twt


df["Cleaned_Tweets"] = df["Tweets"].apply(cleantwt)

df.head()

# get the subjectivity


def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity


def getPolarity(twt):
    return TextBlob(twt).polarity


df["Subjectivity"] = df["Cleaned_Tweets"].apply(getSubjectivity)
df["Polarity"] = df["Cleaned_Tweets"].apply(getPolarity)

df.head()


def getSentiment(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


df["Sentiment"] = df["Polarity"].apply(getSentiment)

df.head()

# Create a scatter plot

plt.figure(figsize=(8, 6))

for i in range(0, df.shape[0]):
    plt.scatter(df["Polarity"][i], color="Purple")

plt.title("Sentiment Analysis")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.show()
