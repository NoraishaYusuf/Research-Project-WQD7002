# =============================================================================
# PREPROCESSING MALAY DATA (test set)
# =============================================================================

#%%
"Execute preprocessing functions"
runfile('C:/Users/shash/RP_functions.py', wdir='C:/Users/shash')

#%%

df_test = pd.read_csv("Malay_testdata(labelled).csv")
print("\n")
print("sample of test data tweets")
print(df_test.text.head(8))
print("\n")
print("Summary of the basic information about this DataFrame and its data:")
print(df_test[['target','sentiment']].describe())
#%%
print("target & sentiment total")
print("\n")
print(df_test.groupby('target').size())
print("\n")
print(pd.crosstab(df_test.target,df_test.sentiment))

#%%
df1 = df_test.copy()
#%%
"PREPROCESSING STEPS"
"NOTE: No stopwords removal, to prep data for sentiment analysis experiments"

"PREPROCESS text"
df1['textclean'] = df1['text'].apply(lambda x:preprocess(x))

"REDUCE elongated words"
df1['textclean'] = df1['textclean'].apply(lambda x:reduce(x))

"REMOVE chinese/other languages after preprocessing"
df1['textclean'] = df1['textclean'].apply(lambda x:char_remove(x))

"STANDARDIZE words that have similar meaning & correcting short form and slang words"
df1['textclean'] = df1['textclean'].apply(lambda x: standardize(x))

"STANDARDIZE key terms related to train"
df1['textclean'] = df1['textclean'].apply(lambda x: standardize2(x))

"STANDARDIZE laughter expression e.g. lol, hahaha, hihihi, hehehe"
df1['textclean'] = df1['textclean'].apply(lambda x: standardize3(x))

"STANDARDIZE additional sentiment expression words"
df1['textclean'] = df1['textclean'].apply(lambda x: standardize_senti(x))

#%%
"SPLIT malay words that starts with x e.g. xnak, xboleh, xnormal"
"LONG RUNTIME !!!"
for row in df1.itertuples():
    %time df1.at[row.Index, 'textclean'] = split_x(row.textclean)
    print('split done for index: ' + str(row.Index))
#%%
df1['textclean'] = df1['textclean'].apply(lambda x: standardize(x))
#%%
"CORRECT spelling"
"LONG RUNTIME !!!"
count = 0
for row in df1.itertuples():
    %time df1.at[row.Index, 'textspell'] = correct(row.textclean)
    count +=1
    print("\n")
    print(count)
    print('spelling done for index: ' + str(row.Index))

#%%
"STEM & LEMMATIZE words"
"LONG RUNTIME !!!"
count = 0
for row in df1.itertuples():
    %time df1.at[row.Index, 'textfin'] = stem_lemma(row.textspell)
    count +=1
    print("\n")
    print('stem & lemmatize done for index: ' + str(row.Index) + " count: " + str(count))
#%%
df1.to_csv("Malay_testdata(labelled_Cleaned).csv",index=False)