# -*- coding: utf-8 -*-
"""IR-Assignment.ipynb

Original file is located at
    https://colab.research.google.com/drive/1e5q18f9ALfhRP-tY6D1QsgMEjACBgcZK

## Import Packages
"""

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import os

## Change this based on your file location

"""## Load Data and Data Preprocessing"""

def loadFileInfo(parentDir):
    '''Loads name of files and vectorizes the file info'''
    folder = os.listdir(parentDir)
    folder.remove('__MACOSX')
    len(folder)
    file_index = dict()
    for i,j in zip(folder,range(1,len(folder)+1)):
        file_index[i] = j
    invertedFileIndex = {value:key for key,value in file_index.items()}
    return file_index, invertedFileIndex

# file_index, invertedFileIndex = loadFileInfo(PARENT)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import 	WordNetLemmatizer

def filterTokens(para):
    '''Toeknizes input strings, removes stop words, removes non alpha numeric characters'''
    stopWords = sorted(stopwords.words('english'))
    tokens = para.replace('\n', ' ').split(' ')
    filteredTokens = list(set([re.sub('[\W\_]','', word.lower()) for word in tokens if word.lower() not in stopWords]))
    t = filteredTokens.pop(0)
    return filteredTokens

def stemming(para):
    '''Stemming of words in input.'''
    stemmer  = PorterStemmer()
    return [stemmer.stem(word) for word in para]

def lemmatize(para):
    '''Lemmatize words in input.'''
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in para]

def genWordDict(file_index, normalize='stem'):
    '''Generates Word Dictionary contianing word indexes from each document. Includes preprocessing steps.'''
    nltk.download('stopwords')
    nltk.download('wordnet')
    Words = dict()
    fileInd = list(file_index.keys())
    print('Reading and Tokenizing files........')
    for i in tqdm(range(len(fileInd))):
        file = fileInd[i]
        f = open(os.path.join(PARENT, file), 'r')
        para = f.read()
        filteredTokens = filterTokens(para)
        normalized = None
        if normalize=='stem':
            normalized = stemming(filteredTokens)
        elif normalize=='lemmatize':
            normalized = lemmatize(filteredTokens)
        Words[file_index[file]] = normalized
    print('\nDone tokenizing and removeing stop words.')
    return Words
    
# docWords = genWordDict(file_index)      # With Stemming
# docWordsStem = docWords     # With Stemming
# docWordsLem = genWordDict(file_index, normalize='lemmatize')    # With Lemmetization

def genInverted(wordDict):
    '''Generates Inverted Word data and final sorted Inverted Index Data structure.'''
    Inv_word = dict()
    e_list = []
    Words = wordDict
    for i in Words.keys():
        Inv_word.update({k:list(e_list) for k in Words[i]})
    for i in Words.keys():
        for j in Words[i]:
            Inv_word[j].append(i)
    Inv_index = dict() 
    for word in Inv_word.keys():
        docs = list(set(Inv_word[word]))
        frequency = len(docs)
        docs.sort()
        Inv_index.update({word:[frequency,docs]})
    return Inv_word, Inv_index

# InvWord, InvIndex = genInverted(docWords)

def viewInvIndex(InvIndex):
    '''Print Inverted Index.'''
    for k, (i,j) in enumerate(InvIndex.items()):
        print(str(i) +' - ' + str(j))

# viewInvIndex(InvIndex)

"""## Spelling Correction - Edit Distance"""

def editDistance(w1, w2):
    '''Returns the edit distance between two words.'''
    n1 = len(w1)
    n2 = len(w2)
    dp = [[0 for i in range(n1+1)] for j in range(2)]

    for i in range(n1+1):
        dp[0][i] = i
    
    for i in range(1, n2+1):
        for j in range(0, n1+1):
            if j==0:
                dp[i%2][j] = i
            elif w1[j-1]==w2[i-1]:
                dp[i%2][j] = dp[(i-1)%2][j-1]
            else:
                dp[i % 2][j] = 1 + min( dp[(i-1)%2][j], min(dp[i%2][j-1], dp[(i-1)%2][j-1]) )

    return dp[n2%2][n1]

# editDistance('mac', 'macbeth')

def getCorrectWords(w1, InvIndex, top_k=10):
    '''Returns list of similar words based on edit distance along with their respective edit distances.'''
    editScores = dict()
    for word in InvIndex.keys():
        editScores[word] = editDistance(word, w1)
    editScoresList = list(editScores.items())
    editScoresList = sorted(editScoresList, key=(lambda x: x[1]))
    return editScoresList[:min(top_k, len(InvIndex.keys()))]

# getCorrectWords('man', InvIndex, 5)





"""## List Intersection"""

class Posting:
    def __init__(self,word,frequency,p_list):
        self.word = word
        self.frequency = frequency
        self.p_list = p_list

"""## K Grams"""

def find_relevant_words(k_gram_index,wild_card_word,k):
  '''Find relevant words'''
  k_grams = []
  new_wild_card_word =  '$' + wild_card_word + '$'
  sub_words = new_wild_card_word.split('*')
  if '$' in sub_words:
    sub_words.remove('$')

  relevant_lists = []
  for word in sub_words:
    k_grams = getKGrams(word,k)
    if len(k_grams) == 0:
      k_grams = getKGrams(word,1)
  
    for k_gram in k_grams:
      relevant_lists.append(k_gram_index[k_gram])
  relevant_words = []
  end = 1
  print(len(relevant_lists))
  relevant_words = relevant_lists[0]
  while(end < len(relevant_lists)):
      relevant_words = np.intersect1d(relevant_words,relevant_lists[end])
      end += 1
  return relevant_words
  
 
def createKGramIndex(InvIndex,k):
  unique_words = list(InvIndex.keys())
  k_gram_index = dict()
  count = 0
  for word in unique_words:
    count += 1
    new_word = '$' + word + '$'
    k_grams = getKGrams(new_word,k)
    for k_gram in k_grams:
      if k_gram in k_gram_index.keys():
        k_gram_index[k_gram].append(word)
      else:
        k_gram_index[k_gram] = []
        k_gram_index[k_gram].append(word)
  
  for word in unique_words:
    count += 1
    new_word = '$' + word + '$'
    k_grams = getKGrams(new_word,1)
    for k_gram in k_grams:
      if k_gram in k_gram_index.keys():
        k_gram_index[k_gram].append(word)
      else:
        k_gram_index[k_gram] = []
        k_gram_index[k_gram].append(word)
  for key in k_gram_index.keys():
    k_gram_index[key].sort()
  return k_gram_index

# k_gram_index = createKGramIndex(InvIndex,2)

def getKGrams(word,k):
  k_grams = []
  length = len(word)
  start = 0
  end = k
  while(end <= length):
    k_grams.append(word[start:end])
    start += 1
    end += 1
  return k_grams

# getKGrams('love',2)



"""## Query"""

def not_list(L,n):
    '''Code for finding out the not boolean operator'''
    not_list = []
    for i in range(1,n+1):
        if i not in L:
            not_list.append(i)
    return not_list

def union(l1,l2):
    '''Union of Two Lists'''
    return l1+l2

#Assuming the spellings are correct and the given the query we can do the folowing analysis
# query = input()
# query = 'brutus and man or caesar'

def get_relevant_docs(query,InvIndex,file_index):
    '''Given a query, it returns a list of all the relevant docs satisfying the query'''
    and_split = list()
    or_split = list()

    def get_or_split(query):
        l_and_splits = []
        l_or_splits = []

        or_split = query.split('or')
        for i,j in zip(or_split,range(0,len(or_split))):
            or_split[j] = i.strip()
        for i in or_split:
            if 'and' in i:
                l_and_splits.append(i)
            else:
                l_or_splits.append(i)
        return l_or_splits, l_and_splits
        
    l_or_splits, l_and_splits = get_or_split(query)

    def get_and_split(L):

        p_and_list= []
        for i in L:
            l = i.split('and')
            for j in l:
                p_and_list.append(j.strip())
        return p_and_list

    p_and_list = get_and_split(l_and_splits)


    #getting all the relevant documents from the corpus of documents
    all_docs_and = []

    for i in p_and_list:
        if 'not' not in i:
            try:
             
                all_docs_and.append(InvIndex[i][1])
            except:
                all_docs_and.append([])
        if 'not' in i:
            try:
             
                i = i.split(' ')[1]
                all_docs_and.append(not_list(InvIndex[i][1],len(file_index)))
            except:
                all_docs_and.append(not_list([],len(file_index)))


    all_docs_or = []
    for i in l_or_splits:
        if 'not' not in i:
            try:
           
                all_docs_or.append(InvIndex[i][1])
            except:
                all_docs_or.append([])
        if 'not' in i:
            try:
              
                i = i.split(' ')[1]
                all_docs_or.append(not_list(InvIndex[i][1],len(file_index)))
            except:
                all_docs_or.append(not_list([],len(file_index)))

    l = []
    almost_lists = []

    for i in range(0,len(all_docs_and)-1,2):
        l = np.intersect1d(all_docs_and[i],all_docs_and[i+1])
        almost_lists.append(l)

    final_lists = []
    for i in almost_lists:
        final_lists.extend(i)
    for i in all_docs_or:
        final_lists.extend(i)

    return list(set(final_lists))

def generate_query(query):
    '''
    Generate proper queries for the given query
    '''
    all_words = query.split(' ')
    new_queries = []
    for i in all_words:
        if i not in ['and', 'or', 'not']:
            if '*' not in i:
                j = getCorrectWords(i,InvIndex, 1)[0][0]
                j = stemming([j])
                new_queries.append(j[0])

            else:
                j = find_relevant_words(k_gram_index,i,2)
                length = len(j)
                ll = int(np.log2(length+1))
                lll = j[0:ll]
                print(lll)
                lll = stemming(lll)
                print(lll)
                new_queries.append(list(lll))
        else:
            new_queries.append(i)
    return new_queries

def search_and_retrieve(query):
    '''Search and retrieve: Get all the docs for the given query post preprocessing'''
    q = generate_query(query)


    queries = []

    # for i in range(len(q)-1):
    #     string = string + new_queries + ' '
    # for j in q[-1]:
    #     string = string +q[-1][]
    if type(q[-1]) == type([]):
        for i in q[-1]:
            nl = q[0:-1]
            nl.append(i)
            query = (' ').join(nl)
            queries.append(query)
    else:
        query = (' ').join(q)
        queries.append(query)
    print(queries)

    d = dict()
    for i in queries:
        d[i] = get_relevant_docs(i,InvIndex,file_index)
    print(d)

