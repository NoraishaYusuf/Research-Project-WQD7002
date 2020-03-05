# =============================================================================
# Sentiment Analysis of Manglish labelled tweets
# =============================================================================
#%%
from pprint import pprint
import numpy as np
from numpy import loadtxt
import pandas as pd
from collections import Counter
import gensim
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost
from sklearn import svm
#%%
df = pd.read_csv("Malay_testdata(labelled_Cleaned)_labelsUpdated.csv")


"Exclude positive sentiment & non-event tweets"
df_no_pos = df[~(df['sentiment']=='positive')]
df_event = df_no_pos[df_no_pos['target']=='yes']
df_event.shape
#%%
"retrieve only event tweet text and sentiment"
df_event = df_event[["textfin","sentiment"]].reset_index(drop=True)
print(df_event.shape)
print("\n")
print(df_event.sample(6))
print("\n")
print("Sentiment size: ")
print(df_event.groupby(['sentiment']).size())
#%%
"Encoding sentiment value"
df_event1 = df_event.copy() 
le = preprocessing.LabelEncoder()
df_event1['sentiment'] = le.fit_transform(df_event1.sentiment.values)

#%%
# =============================================================================
# Word2Vec
# =============================================================================
tokenized_tweet = df_event1['textfin'].apply(lambda x: x.split()) # tokenizing 

model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=800, # desired no. of features/independent variables
            window=10, # context window size
            min_count=2, # Ignores all words with total frequency lower than 2.                                  
            sg = 6, # 1 for skip-gram model
            hs = 0,
            negative = 5, # for negative sampling
            workers= 15, # no.of cores
            iter=100,
            seed = 34
) 
#             
model_w2v.train(tokenized_tweet, total_examples= len(df_event1['textfin']), epochs=10)
#%%
model_w2v.wv.most_similar(positive="problem")
#%%
model_w2v.wv.most_similar(positive="beritahu")
#%%
# measure cosine similarity between two terms
model_w2v.wv.similarity(w1="delay",w2="lambat") 
#%%
# This will give the total number of words in the vocabolary created from this dataset
model_w2v.wv.syn0.shape
#%%
"Feature vectors"

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec
#%%
wordvec_arrays = np.zeros((len(tokenized_tweet),800 )) #features
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 800) #features
wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape # (docs, #features)
#%%
# =============================================================================
# Split training and test set
# =============================================================================
X = df_event1.drop(columns='sentiment', axis=1)
Y = df_event1.sentiment.values

X_w2v = wordvec_df.iloc[X.index,:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=50, test_size=0.30)
print("test count: " + str(Counter(y_test)))
print(Counter(y_train))

xtrain_w2v = wordvec_df.iloc[x_train.index,:]
xtest_w2v = wordvec_df.iloc[x_test.index,:]

#%%
# =============================================================================
# Random Forest classifier
# =============================================================================

#%%
"Tuning RF model"
"Quite LONG RUNTIME !!"
# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 7, 9, 10, 12], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 14], 
              'min_samples_split': [2, 3, 5, 8],
              'min_samples_leaf': [1,5,8,10],
              'class_weight': ["balanced"]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(xtrain_w2v, y_train)
#%%
# Set the clf to the best combination of parameters
rf_clf_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rf_clf_tuned.fit(xtrain_w2v, y_train)
#%%
rf_predictions = rf_clf_tuned.predict(xtest_w2v)
print("\n")
print("accuracy: %f" % accuracy_score(y_test, rf_predictions))
print("\n")
pprint(classification_report(y_test, rf_predictions))
print("\n")
matrix = confusion_matrix(y_test, rf_predictions)
print("confusion matrix : ");print(matrix)


#%%
# =============================================================================
# XgBoost
# =============================================================================

"Tuning XGB model"
"Quite LONG RUNTIME !!"

estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

xgb_grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'accuracy',
    n_jobs = 10,
    cv = 10,
    verbose=True
)

xgb_grid_search.fit(xtrain_w2v, y_train)
#%%
xgb_clf_tuned = xgb_grid_search.best_estimator_
xgb_clf_tuned
#%%
xgb_grid_search.best_score_
#%%
xgb_predictions = xgb_clf_tuned.predict(xtest_w2v)
print("\n")
print("accuracy: %f" % accuracy_score(y_test, xgb_predictions))
print("\n")
pprint(classification_report(y_test, xgb_predictions))
print("\n")
matrix = confusion_matrix(y_test, xgb_predictions)
print("confusion matrix : ");print(matrix)




#%%
# =============================================================================
# SVM
# =============================================================================

"Tuning SVM model"
"Quite LONG RUNTIME !!"

svm_clf = svm.SVC()
parameters = [{'C' : [1000, 10000, 100000], 'probability' : [True], 'kernel' : ['rbf','linear','poly'], 'gamma' : [0.001, 0.003, 0.005, 0.007, 0.009, 0.10, 0.5]}]
grid_search = GridSearchCV(estimator = svm_clf, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_obj = grid_search.fit(xtrain_w2v, y_train)
#%%
grid_obj.best_score_
#%%
# Set the clf to the best combination of parameters
svm_clf_tuned = grid_obj.best_estimator_
svm_clf_tuned
#%%
svm_predictions = svm_clf_tuned.predict(xtest_w2v)
print("\n")
print("accuracy: %f" % accuracy_score(y_test, svm_predictions))
print("\n")
pprint(classification_report(y_test, svm_predictions))
print("\n")
matrix = confusion_matrix(y_test, svm_predictions)
print("confusion matrix : ");print(matrix)

#%%
# =============================================================================
# AUC-ROC to compare classifiers
# =============================================================================

"Computations"
# Instantiate the classfiers and make a list
classifiers = [rf_clf_tuned, 
               xgb_clf_tuned,
               svm_clf_tuned
               ]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(xtrain_w2v, y_train)
    yproba = model.predict_proba(xtest_w2v)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


#%%
"Plot ROC curves"
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


#%%
# =============================================================================
# Boxplots to compare stratified kfold results of classifiers
# =============================================================================

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('RF', rf_clf_tuned))
models.append(('XGB', xgb_clf_tuned))
models.append(('SVM', svm_clf_tuned))

# evaluate each model in turn
kresults = []
names = []
scoring = 'accuracy'
for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_w2v, Y, cv=kfold, scoring=scoring)
    kresults.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#%% 
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Sentiment_Manglish: ML Classifier Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Classifiers')
ax = fig.add_subplot(111)
#plt.boxplot(kresults)
box = plt.boxplot(kresults,vert=0,patch_artist=True, labels=names)
colors = ['lightblue', 'orange', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#ax.set_xticklabels(names)
plt.show()

#%%
# 15.01.2019
from sklearn.externals import joblib
#%%
joblib.dump(wordvec_df, 'wordvec_df_mal.pkl')
#%%
joblib.dump(svm_clf_tuned, 'svm_clf_mal.pkl')
#%%
# Build Text Classification Pipeline
svm_pipeline = make_pipeline(cv, lr)

# save the list of prediction classes (positive, negative)
classes = list(lr_pipeline.classes_)
#%%
from gensim.test.utils import common_texts
from gensim.sklearn_api import W2VTransformer
#%%


# Create a model to represent each word by a 10 dimensional vector.
model = W2VTransformer(size=10, min_count=1, seed=1)

# What is the vector representation of the word 'graph'?
wordvecs = model.fit(common_texts).transform(['graph', 'system'])
assert wordvecs.shape == (2, 10)
#%%
W2V_model_mal = W2VTransformer(size=800, 
                               window=10,
                               min_count=2, 
                               sg = 6,
                               hs = 0,
                               negative = 5,
                               workers = 15,
                               iter = 100,
                               seed=34)
#%%
token_list = tokenized_tweet.to_list()
#%%
wordvecs = W2V_model_mal.fit(token_list)

#%%
# let X be a list of tokenized texts (i.e. list of lists of tokens)
#model = gensim.models.Word2Vec(X, size=100)
"use model_w2v"
"a dictionary of mapping word of 800-dimensional vector"
w2v = dict(zip(model_w2v.wv.index2word, model_w2v.wv.syn0))
#%%
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
#%%

class MeanEmbeddingVectorizer(object):

	def __init__(self, word_model):
		self.word_model = word_model
		self.vector_size = word_model.wv.vector_size

	def fit(self):  # comply with scikit-learn transformer requirement
		return self

	def transform(self, docs):  # comply with scikit-learn transformer requirement
		doc_word_vector = self.word_average_list(docs)
		return doc_word_vector

	def word_average(self, sent):
		"""
		Compute average word vector for a single doc/sentence.
		:param sent: list of sentence tokens
		:return:
			mean: float of averaging word vectors
		"""
		mean = []
		for word in sent:
			if word in self.word_model.wv.vocab:
				mean.append(self.word_model.wv.get_vector(word))

		if not mean:  # empty words
			# If a text is empty, return a vector of zeros.
			logging.warning("cannot compute average owing to no vector for {}".format(sent))
			return np.zeros(self.vector_size)
		else:
			mean = np.array(mean).mean(axis=0)
			return mean


	def word_average_list(self, docs):
		"""
		Compute average word vector for multiple docs, where docs had been tokenized.
		:param docs: list of sentence in list of separated tokens
		:return:
			array of average word vector in shape (len(docs),)
		"""
		return np.vstack([self.word_average(sent) for sent in docs])
#%%
        
    

from gensim.models import Word2Vec, KeyedVectors 
#%%  
model_w2v.wv.save_word2vec_format('modelx.bin', binary=True)
#%%
model = gensim.models.KeyedVectors.load_word2vec_format('modelx.bin', binary=True)
#%%
model["kenapa"]
#%%

mean_vec_tr = MeanEmbeddingVectorizer(model_w2v)
doc_vec = mean_vec_tr.transform(tokenized_tweet)
#%%
svm_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(model_w2v)),
    ("svm", svm_clf_tuned)])

#%%
from sklearn.pipeline import Pipeline
#%%
svm_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(model_w2v)),
    ("svm", svm_clf_tuned)])
#%%
df_event1.sample(3)    
#%%
svm_w2v.predict(tokenized_tweet)
#%%
svm_w2v.predict([["ini", "kenapa", "lama", "sangat", "henti", "dekat", "ampang", "park", "ini", "rosak", "lagi", "ke"]])
#%%
svm_w2v.predict([["dah", "train", "maklum", "sila", "henti"]])

#%%
svm_w2v.predict([tokenized_tweet[217]])

#%%
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#%%
#instantiate CountVectorizer()
cv=CountVectorizer()
 
# this steps generates word counts for the words in your docs
word_count_vector=cv.fit_transform(df_event1['textfin'])

word_count_vector.shape

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
#%%

import nltk

class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    #%%

    #%%
mean_embedding_vectorizer = MeanEmbeddingVectorizer(model_w2v)
mean_embedded = mean_embedding_vectorizer.fit_transform(x_train['textfin'])

#%%
"pipeline"

svm_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(model_w2v)),
    ("svm", svm_clf_tuned)])
    #%%

svm_w2v.predict(["train lambat lagi untuk hari dua",
                 "apa ini jam dah ramai orang dekat Lrt ampang bad service",
                 "selamat pagi line dari sri petaling ke sentul ada problem"])
