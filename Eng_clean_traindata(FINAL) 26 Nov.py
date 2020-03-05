#%%
"Run preprocessing functions"
runfile('C:/Users/shash/RP_functions.py', wdir='C:/Users/shash')

#%%
df = pd.read_csv('rapidkl_1_2018_clean(target).csv')
df = df[df['text']=='ENGLISH']
df.dtypes
df.isna().sum()
df['text'] = df['text'].astype('str') 
df.shape
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
dftren['textclean'] = dftren['textclean'].apply(lambda x: remove_stopwords(x)
#%%
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
dftren['textfin'] = dftren['textfin'].apply(lambda x: standardize_eng(x))

#%%
"FILTER OUT tweets with less than 4 words"
%time dffin = dftren[dftren.textfin.apply(lambda x: word_count(x) >= 4)]
dffin.shape
#%%
dffin.to_csv("Eng_traindata.csv",index=False)
