# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:47:31 2020

@author: alext
"""

import pandas as pd
import numpy as np

import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('RateMyProfessor_Sample data.csv')

ddf = df[(pd.notnull(df.comments)) & (pd.notnull(df.star_rating))]

#%% Prepare text

# Convert to list
data = ddf.comments.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams)

#%% Train predictor of positive or negative rating

ddf['lems'] = [" ".join(doc) for doc in data_lemmatized]

def median_split(r, med):
    if r>=med:
        return 1
    else:
        return 0

ddf['pol'] = ddf.apply(lambda row: median_split(row.star_rating, ddf.star_rating.median()), axis = 1)

X = ddf.lems
y = ddf.pol

vect = CountVectorizer(ngram_range=(1, 2))

X = vect.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = BernoulliNB()

model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

conf = model.predict_proba(X)
print(confusion_matrix(y_test, p_test))

#%% get polarity of text
from textblob import TextBlob
import scipy.stats as stats

ddf['tblob_pol'] = ddf.apply(lambda row: TextBlob(row.lems).sentiment[0], axis = 1)
ddf['nb_conf'] = (model.predict_proba(X)[:,1])*2 - 1

#%% visualize the results
import seaborn as sns
import matplotlib.pyplot as plt
sns.lmplot(x = 'tblob_pol', y = 'star_rating', data = ddf, scatter_kws={"s": 10})
plt.xlabel('Text Polarity')
plt.ylabel('Star Rating')
m, b, r, p, s = stats.linregress(ddf.tblob_pol, ddf.star_rating)