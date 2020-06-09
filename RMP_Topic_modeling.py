# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:46:45 2020

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
stop_words.extend(['comment'])

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

df = pd.read_csv('fvtfjyvw7d-2/RateMyProfessor_Sample data.csv')

ddf = df[(pd.notnull(df.comments)) & (pd.notnull(df.star_rating))]
print(ddf.head())

from gensim.models import CoherenceModel
from pprint import pprint

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
prof_names = list(sent_to_words(ddf.professor_name))
dept_names = list(sent_to_words(ddf.department_name))

p = []
for prof in prof_names:
    for n in prof:
        if n not in p:
            p.append(n)
prof_names = p

d = []
for dept in dept_names:
    for n in dept:
        if n not in d and n != 'department':
            d.append(n)
dept_names = d

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)


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



def remove_prof_names(texts, prof_names = prof_names):
    
    
    tokens = [[token if token not in prof_names else "prof_name" for token in tokens] for tokens in texts]
    return tokens

def remove_dept_names(texts, prof_names = dept_names):
    
    
    tokens = [[token if token not in prof_names else "dept_name" for token in tokens] for tokens in texts]
    return tokens
    

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

data_words_nodepts = remove_dept_names(data_words_nostops)

# Remove prof names
data_words_noprofs = remove_prof_names(data_words_nodepts)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_noprofs)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams)

#%% Create dictionary

data_lemmatized = [[word for word in doc if "-PRON" not in word] for doc in data_lemmatized]
data_lemmatized = [[word for word in doc if "commen" not in word] for doc in data_lemmatized]


id2word = corpora.Dictionary(data_lemmatized)
print(len(id2word))
id2word.filter_extremes()
print(len(id2word))

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

#%%
#convert to tfidf corpus
tfidf = gensim.models.tfidfmodel.TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]
print(tfidf_corpus[:1])

#%% Determine the appropriate number of topics, uncomment to run
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics

#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics

#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.LdaMulticore(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=5,
#                                            passes=10)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values

# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=tfidf_corpus, texts=data_lemmatized, start=2, limit=20, step=2)
# print('done')

# # Show graph
# limit=20; start=2; step=2;
# x = range(start, limit, step)
# print(x)
# print(coherence_values)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()

#%% train LDA model
lda_model = gensim.models.LdaMulticore(corpus=tfidf_corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           passes=10)

pprint(lda_model.print_topics())
doc_lda = lda_model[tfidf_corpus]

#%% determine primary topic for each comment

def prepare_text_for_lda(comment, prof_names = prof_names): 
    
    # Remove Stop Words
    comment_words_nostops = [word for word in simple_preprocess(str(comment)) if word not in stop_words]

    # Remove prof names
    comment_words_noprofs = [token if token not in prof_names else "prof_name" for token in comment_words_nostops]

    # Form Bigrams
    comment_words_bigrams = bigram_mod[comment_words_noprofs]
    doc = nlp(" ".join(comment_words_bigrams)) 
    data_lemmatized = [token.lemma_ for token in doc]
    return data_lemmatized

def get_topics(comment, ldamodel, dictionary = id2word):
    tok = prepare_text_for_lda(comment)
    bow = dictionary.doc2bow(tok)
    #bow = tfidf[bow]
    topics = ldamodel.get_document_topics(bow)
    t = 0
    for tops, perc in topics:
        if perc > t:
            topic = tops
            t = perc
    return topic

ddf['topics'] = ddf.apply(lambda x: get_topics(x.comments, lda_model), axis = 1)
#%% create box plots
import seaborn as sns
gdf = ddf.groupby('topics').median()
sort_order = gdf.star_rating.sort_values(ascending = False).index.to_list()
sns.set_palette(sns.color_palette("RdBu_r", 10))
sns.boxplot(x = 'topics', y = 'star_rating', data = ddf, order = sort_order)
plt.xlabel('Primary Topic')
plt.ylabel('Star Rating')

#%% create word clouds

from wordcloud import WordCloud

for topic in ddf.topics.unique():
    print(topic)
    weights = {pair[0]: pair[1] for pair in lda_model.show_topic(topic, topn=100)}
    
    wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate_from_frequencies(weights)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()