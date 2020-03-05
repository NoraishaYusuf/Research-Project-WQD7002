# =============================================================================
# LDA functions
# =============================================================================
import csv
import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')

import tqdm  

from gensim.models import LdaModel
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint

import pyLDAvis.gensim
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

"stopwords 3!!!"
f = open('stopwords3.txt', 'r')
stopwords3 = f.read().split(", ")
f.close()
#
def remove_stopwords2(sentence):
    newsentence = ""
    tokens = nltk.word_tokenize(sentence)
    words = [word for word in tokens if word not in stopwords3]
    newsentence = " ".join(str(x) for x in words)
    return newsentence

"test to remove station names"

def replace_stn_names(sentence):
    newsentence = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in stationnames),"", sentence)
    nltktokenizer = RegexpTokenizer("[\\w']+|[^\\w\\s]+")
    tokens = nltktokenizer.tokenize(newsentence) 
    return " ".join(str(x) for x in tokens)

"standardize mrt, lrt, and monorail etc."
def standardize4(sentence):  
    newsentence = ""
    wordlist = []
    sentence = str(sentence)
    nltktokenizer = RegexpTokenizer("[\\w']+|[^\\w\\s]+")
    tokens = nltktokenizer.tokenize(sentence) 
    for token in tokens:
        if token in ["Lrt", "Mrt", "Monorail", "train","Rapidkl"]:
            wordlist.append("")              
        else:
            wordlist.append(token)
    newsentence = " ".join(str(x) for x in wordlist)  
    return newsentence

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


"to use lda model to predict new doc"
def pre_new(doc):
#    one = cleaning(doc).split()
    one = doc.split() #remove cleaning
    two = dictionary.doc2bow(one)
    return two

def belong(sentence):
    overall = loading[(pre_new(sentence))]
    return overall

def topic_belong(sentence):
    elements = loading[(pre_new(sentence))]
    n=1
    scores = [x[n] for x in elements]
    return scores


# To produce graph
def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=100, random_state=100)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    return lm_list, c_v

    
def ret_top_model():
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.97:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics

# For tuning - long runtime!!
"For hyperparameter tuning"

def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=50,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

