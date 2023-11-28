import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

# Load the data
data = pd.read_csv('tweets.csv')
#data.head(10)
# Perform sentiment analysis
data['polarity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
tweet_df = data[['text','polarity']]
tweet_df.head(10) 
# Generate a word cloud
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 500, height = 500, background_color ='black',max_words=50, stopwords = stopwords, min_font_size = 7).generate(' '.join(data['text']))
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
