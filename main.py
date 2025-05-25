import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("top10K-TMDB-movies.csv")

print(data.head())

cv = CountVectorizer(max_features = 10000, stop_words='english')
