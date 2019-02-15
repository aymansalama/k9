import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import re
import math
from operator import itemgetter
import spacy
import string
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


# List of stopwords in the nltk lib
en_stopwords = set(stopwords.words('english'))

# Function to tokenize the sentence 
nlp = spacy.load('en')
def clean_comments(text):
    #remove punctuations
    regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
    nopunct = regex.sub(" ", str(text))

    #use spacy to lemmatize comments
    doc = nlp(nopunct.decode('utf8'), disable=['parser','ner'])
    lemma = [token.lemma_ for token in doc]
    filtered_stopwords = []
    for l in lemma:
        if l not in en_stopwords:
            filtered_stopwords.append(l)
    return filtered_stopwords

def tokenizeString(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
    nopunct = regex.sub(" ", str(text))
    token = word_tokenize(nopunct)
    filtered_stopwords = []
    for t in token:
        if t not in en_stopwords:
            filtered_stopwords.append(t)

    filtered_stopwords = [word.lower() for word in filtered_stopwords]
    tok = [w for w in filtered_stopwords if w.isspace() == False]
    return tok


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


def tokenize_input(text):
    # Section to tokenize the input 
    lemmatized = clean_comments(text)
    lemmatized = [word.lower() for word in lemmatized]
    lemmatized_removed_space = [w for w in lemmatized if w.isspace() == False]
    
    return lemmatized_removed_space




#-------------------------------------------------------------------------------
# Section to process ngrams
#-------------------------------------------------------------------------------

def get_bi_string(lemmatized_removed_space):
    bigrams = nltk.collocations.BigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(lemmatized_removed_space)

    # --------------------------------------Bigrams--------------------------------------------------
    bigram_freq = bigramFinder.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
    bigramFreqTable.head().reset_index(drop=True)

    #filter bigrams
    filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

    # format the filtered ngrams into strings 
    try:
        bi_string = []
        for b in filtered_bi['bigram']:
            bi = b[0] + " " + b[1]
            bi_string.append(bi)
    except:
        bi_string = []
    
    return bi_string

def get_tri_string(lemmatized_removed_space):
    trigrams = nltk.collocations.TrigramAssocMeasures()
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(lemmatized_removed_space)

    # --------------------------------------Trigrams--------------------------------------------------
    trigram_freq = trigramFinder.ngram_fd.items()
    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
    trigramFreqTable.head().reset_index(drop=True)

    #filter trigrams
    filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]

    try:
        tri_string = []
        for t in filtered_tri['trigram']:
            tri = t[0] + " " + t[1] + " " + t[2]
            tri_string.append(tri)
    except:
        tri_string = []

    return tri_string





#------------------------------------------------------------------------------
# Section to get the query terms location 
#------------------------------------------------------------------------------

def queryWordsDocuments(list_of_words):
    # Load the words from the wordDb to get all the documents involved with the query words
    with buckets.open('dataStorage/word-DB.txt') as json_file:
        word_db_data = json.load(json_file)
        
        # Nested list of the location of the list of words
        # Eg: albert - [1, 2, 3, 4, 5]
        words_location = []

        for low in list_of_words:
            try:
                words_location.append(word_db_data[low])
            except Exception:
                words_location.append([])
                print("Word not found in db")
                pass

        return words_location

def bigramlocation(bi_string):
    with buckets.open('dataStorage/bigram-cat.txt') as json_file:
        bigram_db_data = json.load(json_file)
        
        # Nested list of the location of the list of words
        # Eg: albert - [1, 2, 3, 4, 5]
        bigram_location = []

        for low in bi_string:
            try:
                bigram_location.append(bigram_db_data[low])
            except Exception:
                bigram_location.append([])
                print("Bigram not found in db")
                pass

        return bigram_location

def trigramlocation(tri_string):
    with buckets.open('dataStorage/trigram-cat.txt') as json_file:
        trigram_db_data = json.load(json_file)
        
        # Nested list of the location of the list of words
        # Eg: albert - [1, 2, 3, 4, 5]
        trigram_location = []

        for low in tri_string:
            try:
                trigram_location.append(trigram_db_data[low])
            except Exception:
                trigram_location.append([])
                print("Trigram not found in db")
                pass

        return trigram_location


# Get the location of single word
def get_all_location(lemmatized_removed_space, bi_string, tri_string):
    all_location = []
    
    word_loc = queryWordsDocuments(lemmatized_removed_space)
    bigram_loc = bigramlocation(bi_string)
    trigram_loc = trigramlocation(tri_string)
    
    all_location.append(word_loc)
    all_location.append(bigram_loc)
    all_location.append(trigram_loc)

    return all_location



#------------------------------------------------------------------------------
# Section to get the query terms frequency in each doc
#------------------------------------------------------------------------------

def wordsFrequencyList(words_location, list_of_words):
    # Load the data from the docs to get the frequency of all the query words
    with buckets.open('dataStorage/data-DB.txt') as data_json:
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
    
    return (total_word_frequency_list, len(page_data))

def bigramFreqList(bigram_location, bigram_list):
    # Load the data from the docs to get the frequency of all the query words
    with buckets.open('dataStorage/ngram-DB.txt') as data_json:
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
    with buckets.open('dataStorage/ngram-DB.txt') as data_json:
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


def get_all_freq(word_loc, lemmatized_removed_space, bigram_loc, bi_string, trigram_loc, tri_string):
    
    all_freq = []
    wordFreq= wordsFrequencyList(word_loc, lemmatized_removed_space)

    if bigram_loc != []:
        bigramFreq = bigramFreqList(bigram_loc, bi_string)
    else:
        bigramFreq = []

    if trigram_loc != []:
        trigramFreq = trigramFreqList(trigram_loc, tri_string)
    else:
        trigramFreq = []

    all_freq.append(wordFreq)
    all_freq.append(bigramFreq)
    all_freq.append(trigramFreq)

    return all_freq





#------------------------------------------------------------------------------
# Section to union all the document's locations
#------------------------------------------------------------------------------

def unionWordsDocuments(words_location):
    # List of all words document
    # All the documents involved in all the queried words
    word_location_union = []
    for word_list in words_location:
        word_location_union = list(set().union(word_location_union, word_list))
    
    return word_location_union


def get_word_doc_union(word_loc, bigram_loc, trigram_loc):
    word_doc_union = unionWordsDocuments(word_loc)

    if bigram_loc != []:
        for b in bigram_loc:
            word_doc_union = list(set().union(word_doc_union, b))

    if trigram_loc != []:
        for t in trigram_loc:
            word_doc_union = list(set().union(word_doc_union, t))

    word_doc_union.sort()

    return word_doc_union


#------------------------------------------------------------------------------
# Section to get frequency in all union docs
#------------------------------------------------------------------------------

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


def get_union_wordFreq(word_loc, word_doc_union, wordFreq, bigram_loc, bigramFreq, trigram_loc, trigramFreq):
    all_union_freq = []
    union_wordFreq = unionWordsFrequencyList(word_loc, word_doc_union, wordFreq)
    union_bigramFreq = []
    union_trigramFreq = []

    if bigram_loc != []:
        union_bigramFreq = unionWordsFrequencyList(bigram_loc, word_doc_union, bigramFreq)

    if trigram_loc != []:
        union_trigramFreq = unionWordsFrequencyList(trigram_loc, word_doc_union, trigramFreq)

    all_union_freq.append(union_wordFreq)
    all_union_freq.append(union_bigramFreq)
    all_union_freq.append(union_trigramFreq)

    return all_union_freq



#------------------------------------------------------------------------------
# TF-IDF
#------------------------------------------------------------------------------

def getNumOfDocWithTerms(word_loc, bigram_loc, trigram_loc):
    all_doc_loc = []
    for w in word_loc:
        all_doc_loc.append(w)
    for b in bigram_loc:
        all_doc_loc.append(b)
    for t in trigram_loc:
        all_doc_loc.append(t)

    length_all_doc_loc = []

    for a in all_doc_loc:
        length_all_doc_loc.append(len(a))
    
    return length_all_doc_loc


def get_idf(word_loc, bigram_loc, trigram_loc, total_num_of_doc):
    num_of_doc_with_terms = getNumOfDocWithTerms(word_loc, bigram_loc, trigram_loc)
    idf = []

    for n in num_of_doc_with_terms:
        if n != 0:
            wtq = float(total_num_of_doc) / n
            idf.append(math.log10(wtq))
        else:
            idf.append(0)
    
    return idf


#-----------------------Calculate the TF----------------------------
# Term frequency is the occurence of the term in a document

def wordCountInDoc(doclist):
    
    wordCount = []
    with open('data-DB.txt') as json_file:
        word_db_file = json.load(json_file)
        
        for d in doclist:
            for w in word_db_file:
                if w["Id"] == d:
                    wordCount.append(w["Total_Word_Count"])
    
    return wordCount


def get_total_tf(union_wordFreq, union_bigramFreq, bigram_loc, union_trigramFreq, trigram_loc):
    total_tf = []
    
    # variable used to determine the length of each word categories such as 
    # single word, bigram and trigram
    # Used as coordination for each word cat
    len_of_word_categories = []

    len_of_word_categories.append(len(union_wordFreq))
    for r in union_wordFreq:
        total_tf.append(r)

    len_of_word_categories.append(len(union_bigramFreq))
    for r in union_bigramFreq:
        total_tf.append(r)

    len_of_word_categories.append(len(union_trigramFreq))
    for t in union_trigramFreq:
        total_tf.append(t)

    return (total_tf, len_of_word_categories)


def get_final_score(idf, total_tf, word_doc_union):
    final_score = []

    # word_amount = total_tf[1][0]
    # bigram_amount = total_tf[1][1]
    # trigram_amount = total_tf[1][2]

    # word_score_rank = 1
    # bigram_score_rank = 2
    # trigram_score_rank = 3

    for df, i in zip(idf, total_tf[0]):
        score = []
        for value in i:
            if value > 0:
                value = 1 + math.log10(value)
            score.append(float(df) * float(value))
        final_score.append(score)


    total_final_score = []
    for i in range(0, len(final_score[0])):
        total_final_score.append(sum(j[i] for j in final_score))

    final_list = zip(word_doc_union, total_final_score)
    final_list = sorted(final_list,key=itemgetter(1), reverse=True)
    
    return final_list

def get_document_with_id(final_score_id):
    with buckets.open("dataStorage/id-DB.txt") as json_file:
        id_doc = json.load(json_file)
    
    return id_doc[str(final_score_id)]


def resultsJSON(final_score, token):
    results = []

    with buckets.open("dataStorage/csv-DB.txt") as json_file:
        csv_data = json.load(json_file)

    for r in final_score:
        # Get the content to find the concordance
        content = csv_data[str(r[0])]['content']
        sentencelist = [s.strip() for s in content.splitlines()]
        newl = ""
        for s in sentencelist:
            if s:
                newl = newl + s + " "
            else:
                newl = newl + " "
        sentencelist = newl.split(".")

        concordance = []
        for c in sentencelist:
            if any(x in c.lower() for x in token):
                concordance.append(c)

        comb = ""
        for g in concordance :
            comb = comb + g.strip() + "... "

        # comb = unicodedata.normalize('NFKD', comb).encode('ascii','ignore')
        
        results.append({
            'doc_id': str(r[0]).encode("utf8"),
            'doc_title': csv_data[str(r[0])]['title'].encode("utf8"),
            'doc_url': csv_data[str(r[0])]['url'].encode("utf8"),
            'doc_content': csv_data[str(r[0])]['content'].encode("utf8"),
            'concordance': comb.encode("utf8"),
            'query_term': token,
        })
    return results

def mainProcess(input):
    # Insert the input from html and tokenize the strings
    lemmatized_removed_space = tokenize_input(input)
    token_without_lemma = tokenizeString(input)
    bi_string = get_bi_string(lemmatized_removed_space)
    tri_string = get_tri_string(lemmatized_removed_space)

    end = time.time()
    print(end -start)
    
    # Get all the locations for words, bigrams and trigrams
    all_location = get_all_location(lemmatized_removed_space, bi_string, tri_string)
    word_loc = all_location[0]
    bigram_loc = all_location[1]
    trigram_loc = all_location[2]


    # Get all the frequencies for words, bigrams and trigrams
    all_freq = get_all_freq(word_loc, lemmatized_removed_space, bigram_loc, bi_string, trigram_loc, tri_string)
    wordFreq = all_freq[0][0]
    bigramFreq = all_freq[1]
    trigramFreq = all_freq[2]
    total_amount_of_doc = all_freq[0][1]

    # Union all the related documents to make matrix column index
    word_doc_union = get_word_doc_union(word_loc, bigram_loc, trigram_loc)
    
    # Get all the union freq based on word_doc_union
    all_union_freq = get_union_wordFreq(word_loc, word_doc_union, wordFreq, bigram_loc, bigramFreq, trigram_loc, trigramFreq)
    union_wordFreq = all_union_freq[0]
    union_bigramFreq = all_union_freq[1]
    union_trigramFreq = all_union_freq[2]

    #-----------------------------------------------------------
    # Calculation part
    #-----------------------------------------------------------
    
    # Get the inverse term frequencies
    idf = get_idf(word_loc, bigram_loc, trigram_loc, total_amount_of_doc)
    # Get the total term frequencies
    total_tf = get_total_tf(union_wordFreq, union_bigramFreq, bigram_loc, union_trigramFreq, trigram_loc)
    # Get the final scoring for all documents based on the inputs
    final_score = get_final_score(idf, total_tf, word_doc_union)
    resultsJson = resultsJSON(final_score, token_without_lemma)
    print(resultsJson)


    # return resultsJson
    

text = "what is the achievement of albert einstein?"
mainProcess(text)