#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 08:11:30 2018

@author: ince
"""
import time
import pandas as pd

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


from urllib.request import urlretrieve
import string
import os

def timeit(method):
    """
    Decorator for timing the execution speed of functions
    :param method: function to be timed.
    :return: decorated function
    """
    def timed(*args, **kw):
        ts = time.clock()
        result = method(*args, **kw)
        te = time.clock()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result

    return timed


class NLP():
    
    def __init__(self):
       self.stemmer = nltk.stem.SnowballStemmer('english')
    
    def stemming(self, s):
        """
        This will perform the stemming over all the strings
        """
        return ' '.join(self.stemmer.stem(token) for token in nltk.word_tokenize(s))
    
    def remove_stopwords(self,s):
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(s)         
        filtered_sentence = [w for w in word_tokens if not w in stop_words]         
        return ' '.join(filtered_sentence)
        
    def remove_punctuation(self,s):
        return s.translate(str.maketrans('','',string.punctuation))
    
    def remove_whitespaces(self, s):
        return s.translate(str.maketrans(string.whitespace, ' '*len(string.whitespace)))
    
    @timeit
    def perform_everything(self, s):
        return self.stemming(self.remove_punctuation(self.remove_whitespaces(self.remove_stopwords(s))))
    
    
class FileHandler():
    
    def __init__(self, download=False):
        pass
    
    @timeit
    def _csv_to_pickle(self, trials = 0):
        """
        This take the csv file and transform it into a pickle object f
        This is because we can handle better the pickles.
        """
        df = pd.DataFrame()
        try:
            df = pd.read_csv('airbnb.csv', sep=',')
            df.to_pickle('airbnb.pkl')
            return "OK"
        except:
            if trials < 2:
                if self._download_csv() == "OK":
                    self._csv_to_pickle(trials = trials +1)
                    return "OK"
                else:
                    return "KO"
            else:
                print("Error in loading the csv file!")
                return "KO"
        print("File csv load with success, saving the pickle...")
        return "OK"
    
    @timeit
    def _csv_to_tsv(self):
        """
        This transform the csv file into a directory containing 18260 files with only one row.
        """
        try:
            df = self._pickle_to_df() #file non found is handled here
            df['description'] = df['description'].apply(str)
            df['description'] = df['description'].apply(NLP().perform_everything)
            df['title'] = df['title'].apply(str)
            df['title'] = df['title'].apply(NLP().perform_everything)
            df.drop(df.columns[0], axis=1, inplace=True)
            df.to_csv('airbnb.tsv', encoding='UTF-8', sep='\t', index= False)
        except Exception as e:
            raise(e)
        return "OK"
    
    @timeit
    def _tsv_to_tsv_docs(self, dirname = 'docs', trials = 0):
        """
        This transform the csv file into a directory containing 18260 files with only one row.
        """
        #this try check if the big tsv file exist, else we create a new one
        try:
            f = open('airbnb.tsv', 'r')
        except:
            self._csv_to_tsv()
            if trials < 1:
                self._tsv_to_tsv_docs(trials=trials+1)
                return "Recurring"
            else:
                return "KO"
        #this check if the dir with the files in exist else we create a new one
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
            
        #core 
        i = 0
        for line in f:
            with open('./'+dirname+'/doc_'+str(i)+'.tsv', 'w') as ftmp:
                ftmp.write(line)
            i += 1
        f.close()
        os.remove('./docs/doc_0.tsv')
        return "OK"
    
    @timeit
    def _pickle_to_df(self, trials = 0): 
        df = pd.DataFrame()
        try:
            df = pd.read_pickle('airbnb.pkl')         
        except:
            if trials < 2:
                if self._csv_to_pickle() == "OK":
                    self._pickle_to_df(trials = trials +1)
                    return "Recurring"
                else:
                    return "KO"
            else:
                print("Errore nel caricamento del file ")
                return "KO"
        if len(df) == 0:
            return self._pickle_to_df()
        else:
            return df
        
    @timeit
    def _download_csv(self):
        url = "https://raw.githubusercontent.com/lrnzgiusti/ADM-HW3-GP2/master/Airbnb_Texas_Rentals.csv"
        urlretrieve(url, "airbnb.csv")
        return "OK"



class SearchEngine:
    """
    This will be a great super-class
    """
    tre = 5
    def __init__(self):
        self.quattro = 5
        
    def test(self):
        print(self.tre, self.quattro)
        
class pino(SearchEngine):
    
    def __init__(self):
        self.testo = 10
    
    
    