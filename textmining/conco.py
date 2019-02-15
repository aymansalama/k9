import re
# import nltk

# def get_all_phases_containing_tar_wrd(target_word, tar_passage, left_margin = 10, right_margin = 10):
#     """
#         Function to get all the phases that contain the target word in a text/passage tar_passage.
#         Workaround to save the output given by nltk Concordance function
         
#         str target_word, str tar_passage int left_margin int right_margin --> list of str
#         left_margin and right_margin allocate the number of words/pununciation before and after target word
#         Left margin will take note of the beginning of the text
#     """
     
#     ## Create list of tokens using nltk function
#     tokens = nltk.word_tokenize(tar_passage)
     
#     ## Create the text of tokens
#     text = nltk.Text(tokens)
 
#     ## Collect all the index or offset position of the target word
#     c = nltk.ConcordanceIndex(text.tokens, key = lambda s: s.lower())
 
#     ## Collect the range of the words that is within the target word by using text.tokens[start;end].
#     ## The map function is use so that when the offset position - the target range < 0, it will be default to zero
#     concordance_txt = ([text.tokens[map(lambda x: x-5 if (x-left_margin)>0 else 0,[offset])[0]:offset+right_margin]
#                         for offset in c.offsets(target_word)])
                         
#     ## join the sentences for each of the target phrase and return it
#     return [''.join([x+' ' for x in con_sub]) for con_sub in concordance_txt]

# raw  = """The little pig saw the wolf climb up on the roof and lit a roaring fire in the fireplace and placed on it a large kettle of water.When the wolf finally found the hole in the chimney he crawled down
#           and KERSPLASH right into that kettle of water and that was the end of his troubles with the big bad wolf.
#           The next day the little pig invited his mother over . She said &amp;amp;quot;You see it is just as I told you. 
#           The way to get along in the world is to do things as well as you can.&amp;amp;quot; Fortunately for that little pig,
#           he learned that lesson. And he just lived happily ever after!"""
# # sent = nltk.word_tokenize(sentence)
# # text = nltk.Text(sent)

# results = get_all_phases_containing_tar_wrd('little pig', raw)
# print(results)

raw  = """The little pig saw the Wolf climb up on the roof and lit a roaring fire in the fireplace and placed on it a large kettle of water. When the wolf finally found the hole in the chimney he crawled down and KERSPLASH right into that kettle of water and that was the end of his troubles with the big bad wolf
          
          
          The next days the little pig invited his mother over She said &amp;amp;quot;You see it is just as I told you. This is another sentence in the line that i want to try to split            
          The way to get along in the world is to do the next things as well as you can.&amp;amp;quot; Fortunately for that little pig, he learned that lesson. And he just lived happily ever after!""" 

# sentencelist = [s.strip() for s in raw.splitlines()]
# newlist = g.split(".") for g in sentencelist #split on periods
# print(newlist)


sentencelist = [s.strip() for s in raw.splitlines()]
newl = ""
for s in sentencelist:
    if s:
        newl = newl + s + " "
    else:
        newl = newl + " "
sentencelist = newl.split(".")

a = ['wolf']
# s = [sentence for sentence in sentencelist if sentence.find("wolf") != -1]
s = []
for c in sentencelist:
    if any(x in c.lower() for x in a):
        s.append(c)

comb = ""
for g in s :
    comb = comb + g.strip() + "... "
print(comb)





