import pandas as pd
import numpy as np
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import re
import math
from operator import itemgetter
import gcs_client


# GCS bucket component
credentials_file = 'K9Bucket-2-b6152a9b63fe.json'
project_name = 'k9bucket-2'

credentials = gcs_client.Credentials(credentials_file)
project = gcs_client.Project(project_name, credentials)

bucket = project.list()[0]

def queryWordsDocuments(list_of_words):
    # Load the words from the wordDb to get all the documents involved with the query words
    with open('word-DB.txt') as json_file:
        word_db_data = json.load(json_file)
        
        # Nested list of the location of the list of words
        # Eg: albert - [1, 2, 3, 4, 5]
        words_location = []

        for low in list_of_words:
            try:
                words_location.append(word_db_data[low])
            except Exception:
                print("Word not found in db")
                pass

        return words_location


def wordsFrequencyList(words_location, list_of_words):
    # Load the data from the docs to get the frequency of all the query words
    with open('data-DB.txt') as data_json:
        page_data = json.load(data_json)

        # List of all word frequency in each documents for all query words
        total_word_frequency_list = []
        for low, wl in zip(list_of_words, words_location):
            # List of word frequency in each related documents
            word_frequency_list = []
            for location in wl:
                for l in page_data:
                    if l["Id"] == location:
                        word_frequency_list.append(l["Word_frequency"][low])
                        break
            total_word_frequency_list.append(word_frequency_list)
    
    return total_word_frequency_list

def unionWordsDocuments(words_location):
    # List of all words document
    # All the documents involved in all the queried words
    word_location_union = []
    for word_list in words_location:
        word_location_union = list(set().union(word_location_union, word_list))
    
    return word_location_union

    
def unionWordsFrequencyList(words_location, word_location_union, total_word_frequency_list):
    # All the word frequency in the union documents
    # word_location_union & total_union_freq will be used together to form a matrix of frequency of each word in each doc
    total_union_freq = []
    for i in range(0, len(words_location)):
        union_freq = []
        for location_union_index in word_location_union:
            if location_union_index in words_location[i]:
                for loc in range(0, len(words_location[i])):
                    if(location_union_index == words_location[i][loc]):
                        union_freq.append(total_word_frequency_list[i][loc])
            else:
                union_freq.append(0)
        # print(union_freq)
        total_union_freq.append(union_freq)
    
    return total_union_freq

en_stopwords = set(stopwords.words('english'))

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

# Function to process the input 
def extract_query_text(text):
    sent_token = sent_tokenize(text)
    punctRemover = RegexpTokenizer(r'\w+')
    filtered_word_token = []

    for s in sent_token:
        removedPunct = punctRemover.tokenize(s.lower())
        for w in removedPunct:
            if w not in en_stopwords:
                filtered_word_token.append(w)
    
    return filtered_word_token

text = "When Jane Austen born?"
filtered_word_token = extract_query_text(text)
words_location = queryWordsDocuments(filtered_word_token)
total_word_frequency_list = wordsFrequencyList(words_location, filtered_word_token)



bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()

bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(filtered_word_token)
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(filtered_word_token)

# --------------------------------------Bigrams--------------------------------------------------
bigram_freq = bigramFinder.ngram_fd.items()
bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
bigramFreqTable.head().reset_index(drop=True)

#filter bigrams
filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]


# --------------------------------------Trigrams--------------------------------------------------
trigram_freq = trigramFinder.ngram_fd.items()
trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
trigramFreqTable.head().reset_index(drop=True)

#filter trigrams
filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]









# format the filtered ngrams into strings     
bi_string = []
for b in filtered_bi['bigram']:
    bi = b[0] + " " + b[1]
    bi_string.append(bi)

tri_string = []
for t in filtered_tri['trigram']:
    tri = t[0] + " " + t[1] + " " + t[2]
    tri_string.append(tri)

def bigramlocation(bi_string):
    with open('bigram-cat.txt') as json_file:
        bigram_db_data = json.load(json_file)
        
        # Nested list of the location of the list of words
        # Eg: albert - [1, 2, 3, 4, 5]
        bigram_location = []

        for low in bi_string:
            try:
                bigram_location.append(bigram_db_data[low])
            except Exception:
                print("Bigram not found in db")
                pass

        return bigram_location

def trigramlocation(tri_string):
    with open('trigram-cat.txt') as json_file:
        trigram_db_data = json.load(json_file)
        
        # Nested list of the location of the list of words
        # Eg: albert - [1, 2, 3, 4, 5]
        trigram_location = []

        for low in tri_string:
            try:
                trigram_location.append(trigram_db_data[low])
            except Exception:
                print("Trigram not found in db")
                pass

        return trigram_location

def bigramFreqList(bigram_location, bigram_list):
    # Load the data from the docs to get the frequency of all the query words
    with open('ngram-DB.txt') as data_json:
        page_data = json.load(data_json)

        # List of all word frequency in each documents for all query words
        total_word_frequency_list = []
        for low, wl in zip(bigram_list, bigram_location):
            # List of word frequency in each related documents
            word_frequency_list = []
            for location in wl:
                for l in page_data:
                    if l["Id"] == location:
                        word_frequency_list.append(l["Bigrams"][low])
                        break
            total_word_frequency_list.append(word_frequency_list)
    
    return total_word_frequency_list

def trigramFreqList(trigram_location, trigram_list):
    # Load the data from the docs to get the frequency of all the query words
    with open('ngram-DB.txt') as data_json:
        page_data = json.load(data_json)

        # List of all word frequency in each documents for all query words
        total_word_frequency_list = []
        for low, wl in zip(trigram_list, trigram_location):
            # List of word frequency in each related documents
            word_frequency_list = []
            for location in wl:
                for l in page_data:
                    if l["Id"] == location:
                        word_frequency_list.append(l["Trigrams"][low])
                        break
            total_word_frequency_list.append(word_frequency_list)
    
    return total_word_frequency_list

def finalUnionFrequencyList(words_location, word_location_union, total_word_frequency_list):
    # All the word frequency in the union documents
    # word_location_union & total_union_freq will be used together to form a matrix of frequency of each word in each doc
    total_union_freq = []
    for i in range(0, len(words_location)):
        union_freq = []
        for location_union_index in word_location_union:
            if location_union_index in words_location[i]:
                for loc in range(0, len(words_location[i])):
                    if(location_union_index == words_location[i][loc]):
                        union_freq.append(total_word_frequency_list[i][loc])
            else:
                union_freq.append(0)
        # print(union_freq)
        total_union_freq.append(union_freq)
    
    return total_union_freq

def wordCountInDoc(doclist):
    
    wordCount = []
    with open('data-DB.txt') as json_file:
        word_db_file = json.load(json_file)
        
        for d in doclist:
            for w in word_db_file:
                if w["Id"] == d:
                    wordCount.append(w["Total_Word_Count"])
    
    return wordCount
            

    
bigram_loc = bigramlocation(bi_string)
trigram_loc = trigramlocation(tri_string)


documents_union = unionWordsDocuments(words_location)

if bigram_loc != []:
    bigram_location_union = []
    for word_list in bigram_loc:
        bigram_location_union = list(set().union(documents_union, word_list))
    
    bigramFreq = bigramFreqList(bigram_loc, bi_string)
else:
    bigram_location_union = documents_union
    bigramUnionFreq = []

if trigram_loc != []:
    trigram_location_union = []
    for word_list in trigram_loc:
        trigram_location_union = list(set().union(bigram_location_union, word_list))
    trigramFreq = trigramFreqList(trigram_loc, tri_string)
    trigramUnionFreq = finalUnionFrequencyList(trigram_loc, trigram_location_union, trigramFreq)
else:
    trigram_location_union = bigram_location_union
    trigramUnionFreq = []

bigramUnionFreq = finalUnionFrequencyList(bigram_loc, trigram_location_union, bigramFreq)




def getNumOfDocWithTerms():
    all_doc_loc = []
    for w in words_location:
        all_doc_loc.append(w)
    for b in bigram_loc:
        all_doc_loc.append(b)
    for t in trigram_loc:
        all_doc_loc.append(t)

    length_all_doc_loc = []
    for a in all_doc_loc:
        length_all_doc_loc.append(len(a))
    
    return length_all_doc_loc




















with open("convert.csv", "r") as csv_file:
    reader = pd.read_csv(csv_file)
    id = reader['Id']

total_num_of_doc = len(id)
num_of_doc_with_terms = getNumOfDocWithTerms()

idf = []

for n in num_of_doc_with_terms:
    wtq = total_num_of_doc / n
    idf.append(math.log10(wtq))




total_union_freq = unionWordsFrequencyList(words_location, trigram_location_union, total_word_frequency_list)
wordCount = wordCountInDoc(trigram_location_union)
print(trigram_location_union)
print('\n')



total_tf = []

for r in total_union_freq:
    print(r)
    print(wordCount)
    tf = []
    for s, w in zip(r, wordCount):
        tf.append(float(s)/w)
    print(tf)
    print('\n\n\n')
    total_tf.append(tf)


if bigramUnionFreq != []:
    for r in bigramUnionFreq:
        print(r)
        tf = []
        for s, w in zip(r, wordCount):
            tf.append(float(s)/w)
        total_tf.append(tf)

if trigramUnionFreq != []:
    for t in trigramUnionFreq:
        print(t)
        tf = []
        for s, w in zip(t, wordCount):
            tf.append(float(s)/w)
        total_tf.append(tf)









final_score = []

for df, i in zip(idf, total_tf):
    score = []
    for value in i:
        score.append(float(df) * value)
    final_score.append(score)


total_final_score = []
for i in range(0, len(final_score[0])):
    total_final_score.append(sum(j[i] for j in final_score))

final_list = zip(trigram_location_union, total_final_score)
final_list = sorted(final_list,key=itemgetter(1))
print(final_list)








    

            




