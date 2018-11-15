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

import re

from urllib.request import urlretrieve
import string
import os


from math import log2

import json
import pickle

from scipy.spatial.distance import cosine

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
        This will perform the snowball stemming over all the strings
        """
        return ' '.join(self.stemmer.stem(token) for token in nltk.word_tokenize(s))

    def remove_stopwords(self, s):
        """
        Remove stopwords
        """
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(s)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)

    def remove_punctuation(self, s):
        return s.translate(str.maketrans('', '', string.punctuation))

    def remove_whitespaces(self, s):
        return s.translate(str.maketrans(string.whitespace, ' '*len(string.whitespace)))

    def perform_everything(self, s):
        return self.remove_string_special_chars(
            self.stemming(
                    self.remove_punctuation(
                            self.remove_whitespaces(
                                    self.remove_stopwords(s)))))

    def remove_string_special_chars(self, s):
        stripped = re.sub(r'[^\w\s]', '', s)
        stripped = re.sub(r'_', ' ', stripped)
        stripped = re.sub(r'\s+', ' ', stripped)
        stripped = stripped.strip()
        return stripped


class FileHandler():

    def __init__(self, download=False):
        pass

    @timeit
    def _csv_to_pickle(self, trials=0):
        """
        This take the csv file and transform it into a pickle object f
        This is because we can handle better the pickles.
        """
        df = pd.DataFrame()
        try:
            df = pd.read_csv('airbnb.csv', sep=',')
            df.to_pickle('airbnb.pkl')
            return "OK"
        except Exception:
            if trials < 2:
                if self._download_csv() == "OK":
                    self._csv_to_pickle(trials=trials +1)
                    return "OK"
                else:
                    return "KO"
        print("File csv load with success, saving the pickle...")
        return "OK"

    @timeit
    def _csv_to_tsv(self):
        """
        This transform the csv file into a big tsv file.
        The big tsv file is a processed version of the csv file
        """
        try:
            df = self._pickle_to_df() #file non found is handled here
            df['description'] = df['description'].apply(str)
            df['description'] = df['description'].apply(NLP().perform_everything)
            df['title'] = df['title'].apply(str)
            df['title'] = df['title'].apply(NLP().perform_everything)
            df.drop(df.columns[0], axis=1, inplace=True)
            df.to_csv('airbnb.tsv', encoding='UTF-8', sep='\t', index=False)
        except Exception:
            return "KO"
        return "OK"

    @timeit
    def _tsv_to_tsv_docs(self, dirname='docs', trials=0):
        """
        This transform the big tsv file into a directory.
        The directory contains about 18260 files with only one row.
        The row is an house in airbnb
        """
        #this try check if the big tsv file exist, else we create a new one
        try:
            f = open('airbnb.tsv', 'r')
        except Exception:
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
    def _pickle_to_df(self, trials=0):
        """
        This loads the pickle file associated with the dataframe.
        """
        df = pd.DataFrame()
        try:
            df = pd.read_pickle('airbnb.pkl')
        except Exception:
            if trials < 2:
                if self._csv_to_pickle() == "OK":
                    self._pickle_to_df(trials=trials +1)
                    return "Recurring"
                else:
                    return "KO"
        if len(df) == 0:
            return self._pickle_to_df()
        else:
            return df

    @timeit
    def _download_csv(self):
        """
        This download the Airbnb cvs file
        """
        url = "https://raw.githubusercontent.com/lrnzgiusti/ADM-HW3-GP2/master/Airbnb_Texas_Rentals.csv"
        urlretrieve(url, "airbnb.csv")
        return "OK"



class Miner:

    def __init__(self):
        self.vocab = self.read_vocab()
        self.inverted_index_1 = self._load_inverted_index_1()
        self.inverted_index_2 = self.tfidf_loader()

    @timeit
    def build_vocab(self):
        """
        This function will build a vocab and save it in json format.
        The vocab contains word in the description and in the title
        """
        vocab = {}
        idx = 1
        for i in range(1,18259):
            with open("./docs/doc_"+str(i)+'.tsv', 'r') as doc:
                desc = doc.read().split('\t')
                desc_title = desc[4].split()+desc[7].split()
                for word in desc_title:
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
        json.dump(vocab, open('vocab.json', 'w', encoding='utf-8'))
        return vocab

    def read_vocab(self):
        """
        This read the vocabolary and return it.
        """
        try:
            return json.load(open('vocab.json', 'r'))
        except Exception:
            self.build_vocab()
            return self.read_vocab()

    def build_inverted_index_normal(self):
        """
        This build the inverted index based on the title and description
        Deprecated: we have a mapreduce version.
        """
        dic = {}
        for i in range(1,18259):
            with open("./docs/doc_"+str(i)+'.tsv', 'r') as doc:
                content = doc.read().split('\t')
                desc_title = content[4].split()+content[7].split()

                for elem in desc_title:
                    term_int = self.vocab[elem]
                    if term_int not in dic:
                        s = "doc_"+str(i)
                        a = set()
                        a.add(s)
                        dic[term_int] = a
                    else:
                        l = dic[term_int]
                        s = "doc_"+str(i)
                        l.add(s)
                        dic[term_int] = l
        return dic

    def _load_inverted_index_1(self):
        """
        This will load the inverted index saved in pickle format.
        """
        with open('inverted_index_1.pkl', 'rb') as p:
            inv_ind = pickle.load(p)
        return inv_ind


    def IDF(self,inverted_index):
        """
        This buils the idf for all the terms in the vocab
        and stores them in a dictionary.
        """
        l = {}
        inverted_vocab = {y:x for x, y in self.vocab.items()}
        for term in inverted_vocab:
            l[term] = log2(18259/len(inverted_index[term]))
        return l


    def TF(self):
        """
        This compute the TF score and return a dictionary.
        The dictionary has {
                            term_id_1: {doc_1: TF_doc_1,
                                     doc_2: TF_doc_2,
                                     doc_j: TF_doc_j},
                            term_id_2: {doc_4: TF_doc_4,
                                     doc_2: TF_doc_2,
                                     doc_j: TF_doc_j}
                           }
        """
        
        dic = {}
        
        
        
        for i in range(1,18259):
            with open("./docs/doc_"+str(i)+'.tsv', 'r') as doc:
                content = doc.read().split('\t')
                desc_title = content[4].split()+content[7].split()

                for elem in desc_title:
                    term_int = self.vocab[elem]
                    if term_int not in dic:
                        dic[term_int] = {"doc_"+str(i): 1/len(desc_title)}

                    else:
                        doc_to_tfs = dic[term_int]
                        s = "doc_"+str(i)
                        if s not in doc_to_tfs:
                            doc_to_tfs[s] = 1/len(desc_title)
                        else:
                            doc_to_tfs.update({s: (doc_to_tfs[s])+(1/len(desc_title))})

                        dic.update({term_int : doc_to_tfs})
            
        return dic


    def TFIDF(self):
        """
        {
        term_1:{doc_1: TFIDF_doc1 ,doc_2: TFIDF_doc2},
        term_2:{doc_6: TFIDF_doc6 ,doc_5: TFIDF_doc5},
        }
        """
        tf_score = self.TF()
        idf_score = self.IDF(self._load_inverted_index_1())
        for term_id in tf_score:
            tf_score[term_id].update(
                    {n: idf_score[term_id] * tf_score[term_id][n] 
                    for n in tf_score[term_id].keys()})

        return tf_score

    def tfidf_saver(self):
        with open('tfidf_score.json', 'w', encoding='utf-8')  as tfidf:
            return json.dump(fp=tfidf, obj=self.TFIDF())

    def tfidf_loader(self):
        try:
            with open('tfidf_score.json', 'r', encoding='utf-8')  as tfidf:
                payload = json.load(fp=tfidf)
                return { int(key): payload[key] for key in payload }
        except FileNotFoundError:
            self.tfidf_saver()
            return self.tfidf_loader()




    def query_tfidf_builder(self):

        query_string = input().split()
        query_string = list(map(NLP().perform_everything, query_string))
        query_string = [self.vocab[key] for key in query_string  if key in self.vocab ]
        return query_string


    def cosine_dist_one(self, query, doc):
        
        cos_dist = []
        for elem in query:
            if doc in self.inverted_index_1[elem]:
                cos_dist.append(self.inverted_index_2[elem][doc])
            else:
                cos_dist.append(0)
            
        if any(cos_dist):
            return 1-cosine(cos_dist, query)
        else:
            return 0

    def cosine_dist_all(self, query):
        """
        The result is a dictionary that contains {doc_1: cos_sim(query, doc_1),
                                                  doc_2: cos_sim(query, doc_2)}
        """
        s = "doc_"
        cosine_score = {}
        for i in range(1,18259):
            cosine_score.update({s+str(i) : self.cosine_dist_one(query, s+str(i))})
            
        return cosine_score