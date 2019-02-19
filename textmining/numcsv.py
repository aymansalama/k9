from __future__ import unicode_literals
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.util import ngrams
import string
import json
import csv
import gcsfs
import gcs_client
import time 

start = time.time()

def getBucket():
    credentials_file = 'K9Bucket-2-b6152a9b63fe.json'
    project_name = 'k9bucket-2'

    credentials = gcs_client.Credentials(credentials_file)
    project = gcs_client.Project(project_name, credentials)

    buckets = project.list()[0]

    return buckets

# Define the gcs bucket
buckets = getBucket()

# export GOOGLE_APPLICATION_CREDENTIALS='K9Bucket-2-b6152a9b63fe.json'

fs = gcsfs.GCSFileSystem(project='k9-bucket-2')
bucket_name = 'k9-bucket-2'
# data_DB_path = bucket_name + '/dataStorage/data-DB.txt'
# word_DB_path = bucket_name + '/dataStorage/word-DB.txt'
filename = "eco_crop_items.csv"


with fs.open(bucket_name + '/crawledDataCSV/' + filename) as json_obj:
   reader = pd.read_csv(json_obj)
   title = reader['title']
   content = reader['content']
   id = reader['id']
   url = reader['url']

def convert_csv_to_json(json_doc):
    if json_doc is None:
        json_doc = {}
        for i, t, c, u in zip(id, title, content, url):
            json_doc[str(i)] = {}
            json_doc[str(i)]["title"] = t
            json_doc[str(i)]["content"] = c
            json_doc[str(i)]["url"] = u
        return json_doc
    else:
        for i, t, c, u in zip(id, title, content, url):
            json_doc[str(i)] = {}
            json_doc[str(i)]["title"] = t
            json_doc[str(i)]["content"] = c
            json_doc[str(i)]["url"] = u
    return json_doc


with buckets.open("dataStorage/csv-DB.txt") as json_file:
    all_csv = None
    try:
        all_csv = json.load(json_file)
    except Exception as e:
        print("got %s on json.load()" % e)


all_csv = convert_csv_to_json(all_csv)

with buckets.open("dataStorage/csv-DB.txt", "w") as json_file:
    json_file.write(json.dumps(all_csv))

# List of stopwords in the nltk lib
en_stopwords = set(stopwords.words('english'))

def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

content = content.astype('str')
content = content.map(lambda x: _removeNonAscii(x))
title = title.astype('str')
title = title.map(lambda x: _removeNonAscii(x))
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}

def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    lang = max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]
    if lang == 'english':
        return True
    else:
        return False


#filter for only english comments
# eng_content=content[content.apply(get_language)]
# eng_title = title[title.apply(get_language)]
# print(eng_title)

def get_pos(treebank_tag):
        if treebank_tag.startswith('J'):
                return "a"
        elif treebank_tag.startswith('V'):
                return "v"
        elif treebank_tag.startswith('N'):
                return "n"
        elif treebank_tag.startswith('R'):
                return "r"
        else:
                return " "


wordnet_lemmatizer = WordNetLemmatizer()

def clean_comment(text):
        regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
        nopunct = regex.sub(" ", str(text))
        word_token = word_tokenize(nopunct)
        filtered_stop = [word for word in word_token if word not in en_stopwords]
        pos_tag = nltk.pos_tag(filtered_stop)
        lemmatized = [] 
        for f in pos_tag:
                pos = get_pos(f[1])
                if pos != " ":
                        lemmatized.append(wordnet_lemmatizer.lemmatize(f[0], pos).encode("utf8"))
                else:
                        lemmatized.append(wordnet_lemmatizer.lemmatize(f[0]))
        return lemmatized




def write_to_json(id, frequency, url, word_db):
    total_terms = 0
    for i in frequency:
        total_terms = total_terms + i[1]
    
    # create dict
    word_frequency = {}
    
    # Insert the data into the dictionary based on keys and values
    for f in frequency:
        word_frequency[f[0]] = f[1]
    

    if word_db is None:
        print("No data")
        word_db = []
        word_db.append({
            'Id': id,
            'Url': url,
            'Total_Word_Count': total_terms,
            'Word_frequency': word_frequency
        })
    else:
        print("Got data")
        word_db.append({
            'Id': id,
            'Url': url,
            'Total_Word_Count': total_terms,
            'Word_frequency': word_frequency
        })
    
    return word_db


# function to categorize the words based on id
def words_category(id, filtered_words, word_db_file):

    for fw in filtered_words:
        # If no data, create json format and insert data
        if word_db_file is None:
            print("No data")
            word_db_file = {}
            word_db_file[fw] = []
            word_db_file[fw].append(id)
            return word_db_file
        # If got data, insert the data
        else:
            print("Got Data")
            # If keyword exists in the database, then append the value 
            if fw in word_db_file:
                word_db_file[fw].append(id)
            # If new keyword, create new array to store value
            else:
                word_db_file[fw] = []
                word_db_file[fw].append(id)

    return word_db_file       

def get_document_id(id, filename, id_db_file):
    
    if id_db_file is None:
        print("Initialize Id file")
        id_db_file = {}
        id_db_file[id] = filename
        return id_db_file
    else:
        print("Append new id")
        id_db_file[id] = filename
    
    return id_db_file









lemmatized = content.map(clean_comment)
lemmatized = lemmatized.map(lambda x: [word.lower() for word in x])
lemmatized_removed_space = []
lemmatized_title = title.map(clean_comment)
lemmatized_title = lemmatized_title.map(lambda x: [word.lower() for word in x])

with buckets.open("dataStorage/id-DB.txt") as json_file:
    id_db = None

    try:
        id_db = json.load(json_file)
    except Exception as e:
        print("got %s on json.load()" % e)

for i in id:
    id_db = get_document_id(i, filename, id_db)

with buckets.open("dataStorage/id-DB.txt", "w") as json_file:
    json_file.write(json.dumps(id_db))
    print("All id(s) are recorded")





# Open tht file to reteive all the datas in the files
with buckets.open("dataStorage/data-DB.txt") as json_file:
    word_db = None
    
    try:
        word_db = json.load(json_file)
    except Exception as e:
        print("got %s on json.load()" % e)
        

with buckets.open("dataStorage/word-DB.txt") as json_file:
    word_cat_db = None
    
    try:
        word_cat_db = json.load(json_file)
    except Exception as e:
        print("got %s on json.load()" % e)

#--------------------------------------------------------------------------

# Compute the word data and word cat data
for word_id, words, url, title in zip(id, lemmatized, url, lemmatized_title):
    words = [w for w in words if w.isspace() == False]
    for t in title:
        words.append(t)
    lemmatized_removed_space.append(words)
    freqDist = FreqDist(words)
    frequency = freqDist.most_common()
    
    word_db = write_to_json(word_id, frequency, url, word_db)
    word_cat_db = words_category(word_id, freqDist, word_cat_db)


with buckets.open("dataStorage/data-DB.txt", "w") as json_file:
    json_file.write(json.dumps(word_db))
    print("data-DB modified")

with buckets.open("dataStorage/word-DB.txt", "w") as json_file:
    json_file.write(json.dumps(word_cat_db))
    print("word-DB modified")












#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or '' in ngram or ' 'in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords:
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False

#function to filter for ADJ/NN trigrams
def rightTypesTri(ngram):
    if '-pron-' in ngram or '' in ngram or ' 'in ngram or '  ' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords:
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False


def write_ngram(id, bigram, bifreq, trigram, trifreq, ngram_file):
    bg = {}
    tg = {}
    
    for b, bf in zip(bigram, bifreq):
        bg[b] = bf
    for t, tf in zip(trigram, trifreq):
        tg[t] = tf

    if ngram_file is None:
        print("No data")
        ngram_file = []
        ngram_file.append({
            'Id': id,
            'Bigrams': bg,
            'Trigrams': tg
        })
        return ngram_file
    else:
        ngram_file.append({
            'Id': id,
            'Bigrams': bg,
            'Trigrams': tg
        })
    
    return ngram_file
    
    
def bigram_cat(id, bigram, bigram_file):

    for b in bigram:
        # If no data, create json format and insert data
        if bigram_file is None:
            print("No data")
            bigram_file = {}
            bigram_file[b] = []
            bigram_file[b].append(id)
            return bigram_file
        # If got data, insert the data
        else:
            print("Got Data")
            # If keyword exists in the database, then append the value 
            if b in bigram_file:
                bigram_file[b].append(id)
            # If new keyword, create new array to store value
            else:
                bigram_file[b] = []
                bigram_file[b].append(id)
    
    return bigram_file

def trigram_cat(id, trigram, trigram_file):

    for t in trigram:
        # If no data, create json format and insert data
        if trigram_file is None:
            print("No data")
            trigram_file = {}
            trigram_file[t] = []
            trigram_file[t].append(id)
            return trigram_file
        # If got data, insert the data
        else:
            print("Got Data")
            # If keyword exists in the database, then append the value 
            if t in trigram_file:
                trigram_file[t].append(id)
            # If new keyword, create new array to store value
            else:
                trigram_file[t] = []
                trigram_file[t].append(id)
       
    return trigram_file


    



# Load the file
with buckets.open('dataStorage/ngram-DB.txt') as json_file:  
    ngram_DB = None

    # Check if the file has data 
    try:
        ngram_DB = json.load(json_file)
        print("ngram-DB opened")
    except Exception as e:
        print("got %s on json.load()" % e)

with buckets.open('dataStorage/bigram-cat.txt') as json_file:
    bigram_DB = None

    # Check if the file has data 
    try:
        bigram_DB = json.load(json_file)
        print("bigram-cat opened")
    except Exception as e:
        print("got %s on json.load()" % e)


with buckets.open('dataStorage/trigram-cat.txt') as json_file:
    trigram_DB = None

    # Check if the file has data 
    try:
        trigram_DB = json.load(json_file)
        print("trigram-cat opened")
    except Exception as e:
        print("got %s on json.load()" % e)
    

unlist_comments = [item for items in lemmatized_removed_space for item in items]

def bigram(text):
        all_bigrams = ngrams(text, 2)
        ngram_list = []
        for ngram in all_bigrams:
            lowered_bigram_tokens = map(lambda token: token.lower(), ngram)
            if any(token not in en_stopwords for token in lowered_bigram_tokens):
                ngram_list.append(' '.join(ngram))
        # return ngram_list 
        freqDist = FreqDist(ngram_list)
        frequency = freqDist.most_common()
        return frequency

def trigram(text):
        all_trigrams = ngrams(text, 3)
        ngram_list = []
        for ngram in all_trigrams:
            lowered_trigram_tokens = map(lambda token: token.lower(), ngram)
            if any(token not in en_stopwords for token in lowered_trigram_tokens):
                ngram_list.append(' '.join(ngram))
        # return ngram_list 
        freqDist = FreqDist(ngram_list)
        frequency = freqDist.most_common()
        return frequency

# for loop to loop bigrams and trigrams for each documents
for ngram_id, words in zip(id, lemmatized_removed_space):

    # bigrams = nltk.collocations.BigramAssocMeasures()
    # trigrams = nltk.collocations.TrigramAssocMeasures()

    # bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(words)
    # trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(words)

    # # --------------------------------------Bigrams--------------------------------------------------
    # bigram_freq = bigramFinder.ngram_fd.items()
    # bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
    # bigramFreqTable.head().reset_index(drop=True)

    # #filter bigrams
    # filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

    # # --------------------------------------Trigrams--------------------------------------------------
    # trigram_freq = trigramFinder.ngram_fd.items()
    # trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
    # trigramFreqTable.head().reset_index(drop=True)

    # #filter trigrams
    # filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]


    # # format the filtered ngrams into strings     
    # bi_string = []
    # for b in filtered_bi['bigram']:
    #     bi = b[0] + " " + b[1]
    #     bi_string.append(bi)

    # tri_string = []
    # for t in filtered_tri['trigram']:
    #     tri = t[0] + " " + t[1] + " " + t[2]
    #     tri_string.append(tri)
    bi = bigram(words)
    bi_string = []
    bi_freq = []
    for b in bi:
        bi_string.append(b[0])
        bi_freq.append(b[1])

    tri = trigram(words)
    tri_string = []
    tri_freq = []
    for t in tri:
        tri_string.append(t[0])
        tri_freq.append(t[1]) 


    # Write the ngram and the frequency into file
    ngram_DB = write_ngram(ngram_id, bi_string, bi_freq, tri_string, tri_freq, ngram_DB)

    bigram_DB = bigram_cat(ngram_id, bi_string, bigram_DB)
    trigram_DB = trigram_cat(ngram_id, tri_string, trigram_DB)



with buckets.open('dataStorage/ngram-DB.txt', 'w') as json_file:
    json_file.write(json.dumps(ngram_DB))
    print("ngram-DB modified")

with buckets.open('dataStorage/bigram-cat.txt', "w") as json_file:
    json_file.write(json.dumps(bigram_DB))
    print("bigram-cat modified")

with buckets.open('dataStorage/trigram-cat.txt', "w") as json_file:
    json_file.write(json.dumps(trigram_DB))
    print("trigram-cat modified")

end = time.time()
print(end - start)
