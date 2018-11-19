#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 08:11:30 2018

@author: ince
"""
import time
import json
import pickle
import string
import os
import heapq
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from urllib.request import urlretrieve
from urllib.error import HTTPError
from math import log10, log
from scipy.spatial.distance import cosine
from geopy import distance, geocoders

import pandas as pd
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

    """
    Text processing
    """
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
    """
    Class for handling csv, tsv, pkl files, also for load dataFrame from an object file
    """
    
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
        except FileNotFoundError:
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
        df = self._pickle_to_df() #file non found is handled here
        df['description'] = df['description'].apply(str)
        df['description'] = df['description'].apply(NLP().perform_everything)
        df['title'] = df['title'].apply(str)
        df['title'] = df['title'].apply(NLP().perform_everything)
        df.drop(df.columns[0], axis=1, inplace=True)
        df.to_csv('airbnb.tsv', encoding='UTF-8', sep='\t', index=False)
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
        except FileNotFoundError:
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
        os.remove('./' + dirname +'/doc_0.tsv')
        return "OK"

    def save_raw_docs(self, dirname='raw_docs'):
        try:
            f = open('airbnb.csv', 'r')
        except FileNotFoundError:
            return "KO"
        #this check if the dir with the files in exist else we create a new one
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        #core
        i = 0
        for line in f:
            with open('./'+dirname+'/doc_'+str(i)+'.csv', 'w') as ftmp:
                ftmp.write('"","average_rate_per_night","bedrooms_count",'+
                           '"city","date_of_listing","description","latitude",'+
                           '"longitude","title","url"\n')
                ftmp.write(line)
            i += 1
        f.close()
        os.remove('./' + dirname + '/doc_0.csv')
        return "OK"

    @timeit
    def _pickle_to_df(self, trials=0):
        """
        This loads the pickle file associated with the dataframe.
        """
        df = pd.DataFrame()
        try:
            df = pd.read_pickle('airbnb.pkl')
        except FileNotFoundError:
            if trials < 2:
                if self._csv_to_pickle() == "OK":
                    self._pickle_to_df(trials=trials +1)
                    return "Recurring"
                else:
                    return "KO"
        if df.empty:
            return self._pickle_to_df()
        else:
            return df

    @timeit
    def _download_csv(self):
        """
        This download the Airbnb cvs file
        """
        url = "https://raw.githubusercontent.com/lrnzgiusti/ADM-HW3-GP2/master/Airbnb_Texas_Rentals.csv"
        try:
            urlretrieve(url, "airbnb.csv")
        except HTTPError:
            return "KO"
        return "OK"

class Miner:
    """
    Controller
    """
    
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
        for i in range(1, 18259):
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
        This read the vocabolary and return it as a python dictionary.
        """
        try:
            return json.load(open('vocab.json', 'r'))
        except FileNotFoundError:
            self.build_vocab()
            return self.read_vocab()

    def build_inverted_index_normal(self):
        """
        This build the inverted index based on the title and description
        Deprecated: we have a mapreduce version.
        """
        dic = {}
        for i in range(1, 18259):
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

    def IDF(self, inverted_index):
        """
        This buils the idf for all the terms in the vocab
        and stores them in a dictionary.
        """
        l = {}
        inverted_vocab = {y:x for x, y in self.vocab.items()}
        for term in inverted_vocab:
            l[term] = log10(18259/len(inverted_index[term]))
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
        for i in range(1, 18259):
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
        This function build the second inverted index based on the tf_idf score.
        The tf_idf score stored is the one of the term in each document.
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
        """
        This method is for compute the inverted index with the tfidf and save is as json file
        """
        with open('tfidf_score.json', 'w', encoding='utf-8')  as tfidf:
            return json.dump(fp=tfidf, obj=self.TFIDF())

    def tfidf_loader(self):
        """
        This method is used for load the inverted index stored before with the tfidf_saver
        """
        try:
            with open('tfidf_score.json', 'r', encoding='utf-8')  as tfidf:
                payload = json.load(fp=tfidf)
                return {int(key): payload[key] for key in payload}
        except FileNotFoundError:
            self.tfidf_saver()
            return self.tfidf_loader()

    def _query_builder(self, s='Insert your query: '):
        """
        This function is used for build the query:
        After the user input the string is processed and returned as a list of indexes
        The indexes are made as the value of each processed element of the query in the vocabulary.
        """
        query_string = input(s).split()
        query_string = list(map(NLP().perform_everything, query_string))
        query_string = [self.vocab[key] for key in query_string  if key in self.vocab]
        return query_string

    def _docs_containing_all_the_query(self, query):
        """
        This function returns a set containing the intersection of the documents
        that contains all the words in the query, the query passed to this funciton is
        already vectorized with respect to our vocab
        """
        omega = set(self.inverted_index_1[query[0]])
        for i in range(1, len(query)):
            omega = omega.intersection(self.inverted_index_1[query[i]])
        return omega


    def conjunctive_result(self, dirname='raw_docs'):
        query = self._query_builder()
        documents_containing_entire_query = self._docs_containing_all_the_query(query)
        df = pd.DataFrame()
        for elem in documents_containing_entire_query:
            df = df.append(pd.read_csv('./'+dirname+'/'+elem+'.csv',
                                       encoding='utf-8'), ignore_index=True, sort=False)
        return df

    def _cosine_dist_one(self, query, doc):
        """
        This function compute the cosine similarity (it was asked for the similarity),
        for the input query and the document passed by parameter
        """
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

    def cosine_dist_all(self, query, clean_zeros=False):
        """
        The result is a dictionary that contains {doc_1: cos_sim(query, doc_1),
                                                  doc_2: cos_sim(query, doc_2)}
        """
        cosine_score = {}
        for i in self._docs_containing_all_the_query(query):
            cosine_dist = self._cosine_dist_one(query, i)
            if clean_zeros == True:
                if cosine_dist != 0:
                    cosine_score.update({i : cosine_dist})
            else:
                cosine_score.update({i : cosine_dist})
        return cosine_score

    def heapify(self, score, k=10):
        """
        Top k scores according to heapsort

        """
        heap = []
        for i in score:
            heapq.heappush(heap, (score[i], i))

        return heapq.nlargest(k, heap)

    def conjunct_ranking_result(self, dirname='raw_docs'):
        query = self._query_builder()
        cos_score = self.cosine_dist_all(query, clean_zeros=True)
        heap_result = self.heapify(cos_score)
        df = pd.DataFrame(columns=['title', 'description', 'city', 'url'])
        for elem in heap_result:
            dftmp = pd.read_csv('./'+dirname+'/'+elem[1]+'.csv',
                                       encoding='utf-8',
                                       usecols=['title', 'description', 'city', 'url']
                                       )
            dftmp['score'] = elem[0]
            df = df.append(dftmp, ignore_index=True, sort=True)
            del dftmp
        return df

    def _score(self, price_max,rooms_max, wante_city_coords, record):
        texas_radius = 1298 #not random, we've calculated it

        def price_getter(actual_price, max_price):
            if actual_price > max_price:
                return (max_price-log(actual_price))/actual_price

            else:
                return (-log(actual_price)+max_price)/max_price

        def rooms_getter(actual_room_number, max_room_number):
            return min(actual_room_number/max_room_number, max_room_number/actual_room_number)



        document_price = int(record['average_rate_per_night'].split('$')[1])
        try:
            document_rooms = int(record['bedrooms_count']) #trasformare studio in 0

        except ValueError:
            document_rooms = 0.5

        price_score = price_getter(document_price, price_max)
        room_score = rooms_getter(document_rooms, rooms_max)

        geo_score = (1-distance.distance(wante_city_coords,
                                         (record.loc['latitude'], record.loc['longitude'])).km/texas_radius)
        return (1/3)*(geo_score+price_score+room_score)

    def get_all_scores(self):
        """
        This method get the score according to the parameters passed by input by the user
        """
        geo_names = geocoders.GeoNames(username='lrnzgiusti')
        #df = #il dataframe che esce fuori dalla query 3.1
        docs_containing_enitre_query = self.conjunctive_result()
        price_max = int(input('What price you want?'))
        room_count = int(input('Hou many rooms you want?'))
        wanted_city = input('In which city you would stay?')
        _, city_coords = geo_names.geocode(wanted_city+', TX', timeout=30) #wanted city è un input
        docs_containing_enitre_query['score'] = docs_containing_enitre_query.apply(
                lambda x: self._score(price_max, room_count, city_coords, x),
                axis=1)
        return docs_containing_enitre_query