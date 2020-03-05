#%%
# =============================================================================
# PREPROCESSING MALAY DATA (train set)
# =============================================================================

#%%
"Execute preprocessing functions"
runfile('C:/Users/shash/RP_functions.py', wdir='C:/Users/shash')
#%%
df2018 = pd.read_csv("rapidkl_2018.csv")
df2018.shape
df2018['text'] = df2018['text'].astype('str') 
df_mal2018 = df2018[df2018['lang']=='MALAY']

df2019 = pd.read_csv("rapidkl_2019.csv")
df2019.shape
df2019['text'] = df2019['text'].astype('str') 
df_mal2019 = df2019[df2019['lang']=='MALAY']

df_mal = pd.concat([df_mal2019,df_mal2018],ignore_index=True)
print(df_mal.dtypes)
print("\n")
print(df_mal.isna().sum())
print("\n")
print(df_mal.head())
print("\n")
print(df_mal.text.head())
#%%
df1 = df_mal.copy()
#%%
"REMOVE admin tweets"

admin =  ['ktm_berhad','aduanMOT','MyRapidKL','AskRapidKL',
          'AsianBuses','AstroRadioNews','MRTMalaysia','APADChannel',
          'Rapidpg','Salak_Selatan', 'KLIAtransit']
df1 = df1[~df1.username.isin(admin)]
df1.shape
#%%
"PREPROCESS text"

df1['textclean'] = df1['text'].apply(lambda x:preprocess(x))
#%%
"REMOVE duplicate text"

df1 = df1.drop_duplicates(['textclean'], keep='last')
df1.shape
#%%
"REDUCE elongated words"
df1['textclean'] = df1['textclean'].apply(lambda x:reduce(x))

"REMOVE chinese/other languages after preprocessing"
df1['textclean'] = df1['textclean'].apply(lambda x:char_remove(x))

"STANDARDIZE words that have similar meaning & correcting short form and slang words"
df1['textclean'] = df1['textclean'].apply(lambda x: standardize(x))

"STANDARDIZE key terms related to train"
df1['textclean'] = df1['textclean'].apply(lambda x: standardize2(x))
#%%
"FILTER to select train related tweets"

term = ["Lrt","lrt","Mrt","mrt","Monorail","monorail","station","train","carriage"]
term.extend(stationnames)

dftren = df1[df1.textclean.str.contains('|'.join(term))]
#%%
"FILTER to EXCLUDE bus, ktm, erl and other non train related tweets"

bus_ktm_erl = ["bus","bas","brt","Brt","driver","drebar","drivers","ktm","ecrl","ecrls","erls","erl","ktm_berhad",
               "ktm berhad","uitm", "puncak alam", "air selangor", "ukm", "dbkl", "upm", 
               "puncak perdana", "rapid penang", "rapidpenang","klia express","kliaexpress"]

dftren = dftren[(dftren.textclean.str.contains(('|'.join(bus_ktm_erl)))==False)]
dftren.shape
#%%
"REMOVE stopwords"
dftren['textclean'] = dftren['textclean'].apply(lambda x: remove_stopwords(x))

"STANDARDIZE laughter expression e.g. lol, hahaha, hihihi, hehehe"
dftren['textclean'] = dftren['textclean'].apply(lambda x: standardize3(x))

"REMOVE stopwords"
dftren['textclean'] = dftren['textclean'].apply(lambda x: remove_stopwords(x))

#%%
"SPLIT malay words that starts with x e.g. xnak, xboleh, xnormal"
"LONG RUNTIME!!!"
for row in dftren.itertuples():
    %time dftren.at[row.Index, 'textclean'] = split_x(row.textclean)
    print('split done for index: ' + str(row.Index))

#%%
"STANDARDIZE and remove stopwords"
dftren['textclean'] = dftren['textclean'].apply(lambda x: standardize(x))
dftren['textclean'] = dftren['textclean'].apply(lambda x: remove_stopwords(x))
#%%
"CORRECT spelling"
"LONG RUNTIME !!!"
count = 0
for row in dftren.itertuples():
    %time dftren.at[row.Index, 'textspell'] = correct(row.textclean)
    count +=1
    print("\n")
    print(count)
    print('spelling done for index: ' + str(row.Index))
#%%
"REMOVE stopwords"
dftren['textspell'] = dftren['textspell'].apply(lambda x: remove_stopwords(x))
#%%
"If required run clear cache"
#malaya.clear_cache('language-detection/multinomial')
#%%
"STEM & LEMMATIZE words"
"LONG RUNTIME !!!"
count = 0
for row in dftren.itertuples():
    %time dftren.at[row.Index, 'textfin'] = stem_lemma(row.textspell)
    count +=1
    print("\n")
    print('stem & lemmatize done for index: ' + str(row.Index) + " count: " + str(count))

#%%
"FILTER OUT tweets with less than 4 words"
%time dffin = dftren[dftren.textfin.apply(lambda x: word_count(x) >= 4)]
dffin.shape
#%%  
dffin.to_csv("Malay_traindata.csv",index=False)
