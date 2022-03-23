from nltk.corpus import stopwords
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import 	WordNetLemmatizer
import sys

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
  
def main():
  args = sys.argv
  if args[0]=='-stem':
    stemming(args[1:])
  elif args[0]=='-lemmatize':
    lemmatize(args[1:])
  else:
    filterTokens(args)
