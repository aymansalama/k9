import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.text import Text
from nltk.probability import FreqDist
import json
import re

data = {}

# create list
data['Html'] = []

def read_csv():
    with open("test.csv", "r") as csv_file:
        reader = pd.read_csv(csv_file)
        queryString = "love"
        a = []

        # The query result will return the whole row of the related query
        # The query is based on the substring that exists in the string
        queryResult = reader.loc[reader["Url"].str.contains(queryString, regex=False)]
        length_of_queryResult = len(queryResult) #get the length of the query result

        for i in range(0, length_of_queryResult):
            a.append(queryResult.iloc[i, 1])
        
    csv_file.close()    

# Extract raw content from the csv file 
def extract_content():
    with open("convert.csv", "r") as csv_file:
        reader = pd.read_csv(csv_file)
        id = []
        content = []
        url = []

        # Access the columns by index
        for index, df in reader.iterrows():
            id.append(df['Id'])
            content.append(df['Content'])
            url.append(df['Url'])

        ## Get all the row of the content
        # contentSet = reader.iloc[1, 1]
        # sentenceSet = sent_tokenize(contentSet.decode('utf-8'))
        # punctRemover = RegexpTokenizer(r'\w+')
        # stop_words = set(stopwords.words("english"))
        # filtered_words = []
        # properWordsSet = []
        # properWordsPattern = re.compile('^[a-z\d]+$') #regex to find all the proper words

        # for sent in sentenceSet:
        #     removedPunct = punctRemover.tokenize(sent.lower())
        #     for rp in removedPunct:
        #         if properWordsPattern.findall(rp):
        #             properWordsSet.append(rp)
                
        #     removedChar = [i for i in properWordsSet if len(i) > 1]

        #     for words in removedChar:
        #         if words not in stop_words:
        #             filtered_words.append(words)
        
        # freqDist = FreqDist(filtered_words)
        # words_category(freqDist)

        
        for id, content, url in zip(id, content, url):
            sentenceSet = sent_tokenize(content.decode('utf-8'))
            punctRemover = RegexpTokenizer(r'\w+')
            stop_words = set(stopwords.words("english"))
            filtered_words = []
            properWordsSet = []
            properWordsPattern = re.compile('^[a-z\d]+$') #regex to find all the proper words

            for sent in sentenceSet:
                removedPunct = punctRemover.tokenize(sent.lower())
                for rp in removedPunct:
                    if properWordsPattern.findall(rp):
                        properWordsSet.append(rp)
                    
                removedChar = [i for i in properWordsSet if len(i) > 1]
    
                for words in removedChar:
                    if words not in stop_words:
                        filtered_words.append(words)
                
            freqDist = FreqDist(filtered_words)
            frequency = freqDist.most_common()

            # write_to_json(id, frequency, url)
            words_category(id, freqDist)

def write_to_json(id, frequency, url):

    page_db = None

    with open('data-2.txt') as page_json:
        try:
            page_db = json.load(page_json)
        except Exception as e:
            print("got %s on json.load()" % e)
    
    # create dict
    word_frequency = {}
    
    # Insert the data into the dictionary based on keys and values
    for f in frequency:
        word_frequency[f[0]] = f[1]
    
    if page_db is None:
        page_db = []
        page_db.append({
            'Id': id,
            'Url': url,
            'Word_frequency': word_frequency
        })
    else:
        page_db.append({
            'Id': id,
            'Url': url,
            'Word_frequency': word_frequency
        })

    # append data to the json
    with open('data-2.txt', 'w') as outfile:  
        json.dump(page_db, outfile)



# function to categorize the words based on id
def words_category(id, filtered_words):

    word_db_file = None

    # Load the file
    with open('wordDB.txt') as json_file:    
        
        # Check if the file has data 
        try:
            word_db_file = json.load(json_file)
        except Exception as e:
            print("got %s on json.load()" % e)
        
    for fw in filtered_words:
        # If no data, create json format and insert data
        if word_db_file is None:
            word_db_file = {}
            word_db_file[fw] = []
            word_db_file[fw].append(id)
            with open('wordDB.txt', 'w') as json_file:
                json.dump(word_db_file, json_file)
            json_file.close()
        # If got data, insert the data
        else:
            # If keyword exists in the database, then append the value 
            if fw in word_db_file:
                word_db_file[fw].append(id)
            # If new keyword, create new array to store value
            else:
                word_db_file[fw] = []
                word_db_file[fw].append(id)
            
    # Write data to file
    with open('wordDB.txt', 'w') as json_file:
        json.dump(word_db_file, json_file)
    json_file.close()

extract_content()