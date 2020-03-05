
# =============================================================================
# IMPORT RELEVANT MODULES & RUN FUNCTIONS
# =============================================================================

import pandas as pd
import malaya
import symspellpy
from symspellpy.symspellpy import SymSpell, Verbosity 
import pkg_resources
import string
import re
import nltk
import contractions
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk.download('wordnet')
#nltk.download('omw')
#import timeit
# =============================================================================
# for spelling correction function
# =============================================================================
#Create a new instance of SymSpell. initial_capacity from the original code is omitted since python cannot preallocate memory
symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7,count_threshold=1, compact_level=5)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
   # term_index is the column of the term and count_index is the
   # column of the term frequency
symspell.load_dictionary(dictionary_path, term_index=0,count_index=1)
symspell.load_bigram_dictionary(bigram_path, term_index=0,count_index=2)
max_edit_distance_lookup = 2


# =============================================================================
# Basic Pre-processing function
# =============================================================================

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def preprocess(sentence):
#    sentence = str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")  #remove html tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence) #search and replace html symbols
    rem_RT = re.sub(r'^RT[\s]+', '', cleantext)  # remove mentions
#    rem_tag = re.sub(r'@[A-Za-z0-9]+', '', rem_RT)
    rem_tag = re.sub(r'@(\S+)', '', rem_RT)
#    rem_hash = re.sub(r'#[A-Za-z0-9]+', '', rem_tag)
    rem_hash = re.sub(r'#(\S+)', '', rem_tag)  
    rem_url=re.sub(r'http\S+', '',rem_hash)
    rem_pic_url = re.sub(r'pic.twitter.com\S+', '', rem_url)
    rem_pic_url1 = re.sub(r'twitter.com\S+', '', rem_pic_url)
    rem_frac = rem_pic_url1.replace(u"½", u".5")
    rem_num = re.sub('[0-9]+', '', rem_frac)
    rep_contracts = replace_contractions(rem_num)
    rem_contracts1 = re.sub(r"(\w+)'s",r'\1',rep_contracts) 
    rem_punc = re.sub(r'[^\w\s]',' ',rem_contracts1)
#    rem_punc = re.sub(r"[^\w\d'\s]+",' ',rem_num) #replace with space bcoz words (b4 & after period/dot) will merge
    nltktokenizer = RegexpTokenizer("[\\w']+|[^\\w\\s]+")
    tokens = nltktokenizer.tokenize(rem_punc)
#    words = [word for word in tokens if word not in stopwords_fin]
    return " ".join(tokens)

def reduce(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text) 

def char_remove(sentence):
    return ''.join(word for word in sentence if word in string.printable)
 

# To remove tweets which has < 4 words
def word_count(sentence):
    tokens = nltk.word_tokenize(sentence)
    n_tokens = len(tokens)
    return n_tokens     

# =============================================================================
#  stopwords & special words
# =============================================================================
"Station names"
f = open('stationnames.txt', 'r')
stationnames = f.read().split(", ")
f.close

"Special words/terms"
f = open('specialwords.txt', 'r')
specialwords = f.read().split(", ")
f.close

stationnames_ex = " ".join(stationnames)
token3 = nltk.word_tokenize(stationnames_ex)
uniq3 = list(set(token3))
specialwords.extend(uniq3)

"manglish stopwords"
f = open('stopwords_final1.txt', 'r')
manglish_stopwords = f.read().split(", ")
f.close()

"nltk english stopwords"
stopwords.fileids() # indon lang present
eng_stopwords = nltk.corpus.stopwords.words('english')

stopwords_fin = []
stopwords_fin.extend(manglish_stopwords)
stopwords_fin.extend(eng_stopwords)
print('Number of stop words: %d' % len(stopwords_fin))

not_stopwords = {'time','from', 'where', 'again', 'after', 'through', 'because', 
                 'which', 'who', 'when', 'what', 'to', 'but', 'until', 'out'
                 'how', 'why', 'in', 'then', 'here', 'all', 'any',
                 'few', 'more', 'most', 'other', 'some', 'no', 'not', 
                 'only', 'same', 'too', 'now', 'during', 'for', 'at', 'of' } 
stopwords_fin = list(set([word for word in stopwords_fin if word not in not_stopwords]))

print('Number of stop words final: %d' % len(stopwords_fin))

def remove_stopwords(sentence):
    newsentence = ""
    tokens = nltk.word_tokenize(sentence)
    words = [word for word in tokens if word not in stopwords_fin]
    newsentence = " ".join(str(x) for x in words)
    return newsentence


replacements = {}
with open("replacements2.txt","r") as f:
    for line in f:
        (key, val) = line.strip().split(', ')
        replacements[str(key)] = str(val)
f.close()

standardwords = []
rep_words = []
with open("replacements2.txt","r") as f:
    for line in f:
        (key, val) = line.strip().split(', ')
        rep_words.append(str(val))
f.close()

res1 = " ".join(rep_words)
token1 = nltk.word_tokenize(res1)
uniq1 = list(set(token1))
standardwords.extend(uniq1)

res2 = " ".join(specialwords)
token2 = nltk.word_tokenize(res2)
uniq2 = list(set(token2))
standardwords.extend(uniq2)


def replace(match):
    return replacements[match.group(0)]

def standardize(sentence):
    newsentence = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replacements),
             replace, sentence)
    return newsentence


def standardize2(sentence):  
    newsentence = ""
    wordlist = []
    sentence = str(sentence)
    nltktokenizer = RegexpTokenizer("[\\w']+|[^\\w\\s]+")
    tokens = nltktokenizer.tokenize(sentence) 
    for token in tokens:
        if token in ["hentian", "st", "stn", "stesen", "stesyen", "stsn", "stsyn", "stesn"]:
            wordlist.append("station")
        elif token in ["rel", "rail", "keretapi", "tren", "trin"]:
            wordlist.append("train")
        elif token in ["koc", "gerabak", "gerabk", "grbk", "coach"]:
            wordlist.append("carriage")
        elif token in ["lrts", "mrts", "mrls", "brts", "monorels", "monorails"]:
            wordlist.append(token[:-1])              
        elif token in ["monorel", "monorail", "mrl"]:
            wordlist.append("Monorail")
        elif token in ["rapidkl", "myrapid", "myrapidkl","rpidkl"]:
            wordlist.append("Rapidkl")
        elif token == "bas":
            wordlist.append("bus")         
        elif token in ["lrt", "mrt", "mrl", "brt", "monorail"]:
            wordlist.append(token.title())
        else:
            wordlist.append(token)
    newsentence = " ".join(str(x) for x in wordlist)  
    return newsentence


"REMOVE stopwords first!!"
def standardize3(sentence):
#    shorten = reduce_lengthening(sentence)
    laugh1 = re.sub(r"\b(?:[aeiou]{0,2}h{1,2}[aeiou]{0,2}){2,}h?\b",'laugh',sentence)
    laugh2 = re.sub(r"\b(?:[o]{0,2}l{1,2}[o]{0,2}){2,}l?\b",'laugh',laugh1)
    laugh3 = re.sub(r"\b(?:[k]{1,2}[aeiou]{0,2}[h]{0,2}[aeiou]{0,2}){2,}\b",'laugh', laugh2)
    nltktokenizer = RegexpTokenizer("[\\w']+|[^\\w\\s]+")
    tokens = nltktokenizer.tokenize(laugh3)
    return " ".join(tokens)
"REMOVE stopwords after too!!"


"additional replacement for english data"
replacements_eng = {}
with open("replacements_eng.txt","r") as f:
    for line in f:
        (key, val) = line.strip().split(', ')
        replacements_eng[str(key)] = str(val)
f.close()

def replace_eng(match):
    return replacements_eng[match.group(0)]

def standardize_eng(sentence):
    newsentence = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replacements_eng),
             replace_eng, sentence)
    return newsentence



"additional replacement for sentiment data"
replacements3 = {}
with open("extra_replace_for_sentiment.txt","r") as f:
    for line in f:
        (key, val) = line.strip().split(', ')
        replacements3[str(key)] = str(val)
f.close()

def replace3(match):
    return replacements3[match.group(0)]

def standardize_senti(sentence):
    newsentence = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replacements3),
             replace3, sentence)
    return newsentence
# =============================================================================
# Spelling correction function
# =============================================================================
model = malaya.pos.transformer(model = 'xlnet', size = 'base')
corrector = malaya.spell.probability()
normalizer = malaya.normalize.spell(corrector)

def correct(sentence):
    newsentence = ""
#    pos_dict = dict(model.predict(sentence))
    wordlist = []
    sentence = str(sentence)
    nltktokenizer = RegexpTokenizer("[\\w']+|[^\\w\\s]+")
    tokens = nltktokenizer.tokenize(sentence) 
    multinomial = malaya.language_detection.multinomial()
    for token in tokens:
#        lang = multinomial.predict(token)
        if token in specialwords:wordlist.append(token)
        elif model.predict(token)[0][1] == 'PROPN':wordlist.append(token)
        elif multinomial.predict(token) == 'ENGLISH':
            ns = symspell.lookup(token,'CLOSEST')
            try:
                newtoken = ("{}".format(ns[0].term))
                wordlist.append(newtoken)
            except (IndexError, ValueError):
                wordlist.append(token)
        else:
            n = normalizer.normalize(token)
            newtoken = (list(n.values())[0])
#            if model.predict(token)[0][1] == 'PROPN':
#                wordlist.append(token)
            if newtoken == '' or newtoken is None:
                wordlist.append(token)
            else:
                wordlist.append(newtoken)
#    words = [word for word in wordlist if word not in stopwords_fin]
#    newsentence = " ".join(str(x) for x in wordlist)  
    newsentence = " ".join(str(x) for x in wordlist)
    print("sentence: " + str(sentence) +"\n")
    print("new sentence: " + str(newsentence))    
    return newsentence


def split_x(sentence): 
    res_words = []
    sentence = str(sentence)
    tokens = nltk.word_tokenize(sentence)
    multinomial = malaya.language_detection.multinomial()
    for token in tokens:
        if not token.startswith('x'):
            res_words.append(token)
        else:# token.startswith('x'):
            lang = multinomial.predict(token)
            if token in specialwords:
                res_words.append(token)
            elif lang != 'ENGLISH':
                pattern = re.split(r'\b(x)',token)
                print(token,pattern)
                res_words.extend(pattern)
            else:
                res_words.append(token)
#    xses = pattern.findall(sentence)
#    newsentence = " ".join(str(x) for x in wordlist)
    joined = " ".join(res_words)
    reset = nltk.word_tokenize(joined)
    return " ".join(reset)


lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_eng(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:            
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

def stem_lemma(sentence):
    res_words = []
    newsentence = ""
    tokens = nltk.word_tokenize(sentence)
    multinomial = malaya.language_detection.multinomial()
    for token in tokens:
        lang = multinomial.predict(token)
        #        doc = nlp(token)
#        lang = list(doc._.language.values())[0]
        print("( "+ token + ", " + lang +" )")
        if token in standardwords:
            res_words.append(token)
        elif lang == 'ENGLISH':
            res_words.append(lemmatize_eng(token))
        else:
            res_words.append(malaya.stem.sastrawi(token))
    print("\n")        
    print("sentence: " + str(sentence) +"\n") 
    newsentence = " ".join(res_words)
    print("new sentence: " + str(newsentence)) 
    return newsentence      

