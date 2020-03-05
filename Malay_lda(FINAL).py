# =============================================================================
# MALAY CORPUS: 
# 1. Tuned LDA model for train data
# 2. Use tuned lda model to infer topic distribution on test data
# =============================================================================

"Execute preprocessing functions"
runfile('C:/Users/shash/RP_LDA_functions.py', wdir='C:/Users/shash')
runfile('C:/Users/shash/RP_functions.py', wdir='C:/Users/shash')
#%%
dfmal = pd.read_csv("Malay_traindata.csv")

#%%
"Anonymize train terms e.g. monorail, rapidkl etc."
dfmal['textfin'] = dfmal['textfin'].apply(lambda x: standardize4(x))

"Remove prominent stop words"
dfmal['textfin'] = dfmal['textfin'].apply(lambda x: remove_stopwords2(x))

"Anonymize station names"
dfmal['textfin'] = dfmal['textfin'].apply(lambda x: replace_stn_names(x))



#%%
"PREP elements for LDA: tokenized text, dictionary and vectorized corpus"
corpus_mal = dfmal['textfin'].tolist()

train_texts = [doc.split(" ") for doc in corpus_mal]

bigram = gensim.models.Phrases(train_texts, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[train_texts], threshold=100)  
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

data_words_bigrams = make_bigrams(train_texts)
texts = data_words_bigrams

dictionary = Dictionary(texts)
print("\n")
print("total terms in dictionary")
print(len(dictionary.values()))

corpus = [dictionary.doc2bow(text) for text in texts]

#%%
"plot optimization graph for basic lda model"
"LONG RUNTIME!!"
lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=texts, limit=10)

#%%
"Run pre tuning"
#ldamodel = LdaModel(corpus=corpus, num_topics=4, id2word=dictionary, passes=100,random_state=100)

#%%
"Run post tuning "
ldamodel = gensim.models.LdaMulticore(corpus=corpus, num_topics=8, 
                                        id2word=dictionary,
                                        random_state=100,
                                        chunksize=100,
                                        passes=50,
                                        alpha="asymmetric",
                                        eta=0.91)


# Compute Coherence & perplexity Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)
print('\nPerplexity:', ldamodel.log_perplexity(corpus))

"n=6, stopwords2 removed, full ano, alpha=asymmetric, eta=0.91, coherence:0.4075572   , perplex: -7.11937651  "
"n=8, stopwords2 removed, full ano, alpha=asymmetric, eta=0.91, coherence:0.4403929   , perplex: -7.16848867 "
"n=8, stopwords2 removed, full ano, alpha=0.61, eta=0.91, coherence:0.4172946   , perplex: -7.422954886 "

#%%
"SAVE lda multicore"
ldamodel.save('Malay_tunedLDA(FINAL).model')

#%%
"Load saved model"
loading = LdaModel.load('Malay_tunedLDA(FINAL).model')

#%%
"VIEW & SAVE top terms of topics of lda model"
stored = loading.print_topics(num_words=30)

pprint(stored)

with open("Malay_lda_topictermList.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(stored)
#%%
"create & save pyLDAvis"

prepared_vis_dat = pyLDAvis.gensim.prepare(loading, corpus, dictionary)
#pyLDAvis.show(prepared_vis_dat)
pyLDAvis.save_html(prepared_vis_dat,'Malay_pyLDAvis(FINAL).html')

#%%

"Infering our tuned lda model on test data"

df_test_mal = pd.read_csv("Malay_testdata(labelled_Cleaned).csv") # all data
df_test_mal['LDAtotal']=df_test_mal['textfin'].apply(lambda x: belong(x))
g = (lambda x: pd.Series(topic_belong(x)))
df_test_mal[["topic1","topic2","topic3","topic4","topic5","topic6","topic7","topic8"]] = df_test_mal['textfin'].apply(g)
df_test_mal.to_csv("Malay_test_LDATopics_infer_V2(FINAL).csv", index=False)

#%%

dfmal.to_csv("Malay_traindata_Ano.csv", index=False)
#%%
df_tren = pd.read_csv("Malay_traindata_Ano.csv")

#%%
df_tren['textfin'] = df_tren['textfin'].astype(str)
df_tren['LDAtotal']=df_tren['textfin'].apply(lambda x: belong(x))
g = (lambda x: pd.Series(topic_belong(x)))
df_tren[["topic1","topic2","topic3","topic4","topic5","topic6","topic7","topic8"]] = df_tren['textfin'].apply(g)
df_tren.to_csv("Malay_traindata_Ano_infer(cloud).csv", index=False)


#%%

"TUNING HYPERPARAMETER OF LDA MULTICORE MODEL"
"SUPER LONG RUNTIME!!"

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 8
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75), 
               corpus]

corpus_title = ['100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=dictionary, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results_MAL.csv', index=False)
    pbar.close()
