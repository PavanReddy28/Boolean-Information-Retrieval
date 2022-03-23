import lemmatize, filterTokens, stemming from ./preprocess.py
import os
import nltk
import tqdm
import sys

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

def loadFileInfo(parentDir):
    '''
    Loads name of files and vectorizes the file info
    '''
    folder = os.listdir(parentDir)
    folder.remove('__MACOSX')
    len(folder)
    file_index = dict()
    for i,j in zip(folder,range(1,len(folder)+1)):
        file_index[i] = j
    invertedFileIndex = {value:key for key,value in file_index.items()}
    return file_index, invertedFileIndex
  
def main():
  args = sys.argv
  if args[0]=='-d':
    file_index, invertedFileIndex = loadFileInfo(args[1])
    docWords = genWordDict(file_index)      # With Stemming
    docWordsLem = genWordDict(file_index, normalize='lemmatize')    # With Lemmetization
    
  
  
