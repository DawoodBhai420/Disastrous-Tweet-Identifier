import pandas as pd
df = pd.read_csv('Dataset/train.csv')
id = df["id"]
y = df["target"]
tweets = df["text"]

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
corpus = []
for i in range(len(tweets)):
    tweet = re.sub('[^a-zA-Z]', ' ', tweets[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
classifiers = [GaussianNB(), MultinomialNB(), BernoulliNB()]

max = 0
for clf in classifiers:
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f1 = f1_score(y_test, pred)
    print(f1)
    if f1 > max:
        max = f1
        best_clf = clf

print("f1 score of best classifier: ",max)

# predicting test dataset
df_test = pd.read_csv('Dataset/test.csv')
test_id = df_test.id
test_tweets = df_test.text

test_corpus = []
for i in range(len(test_tweets)):
    tweet = re.sub('[^a-zA-Z]', ' ', tweets[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    test_corpus.append(tweet)

test_X = tfidf.transform(test_corpus).toarray()
pred_y = best_clf.predict(test_X)

#retraining the model on full training dataset
final_clf = MultinomialNB()
final_clf.fit(X,y)
final_pred = final_clf.predict(test_X)

submission = pd.DataFrame({"id":test_id,
                           "target":final_pred
                           })
submission.to_csv('submission.csv', index=False)