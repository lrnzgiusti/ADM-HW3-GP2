#!/usr/bin/env python

import sys
import os
import json
"""
mapper: 
The input is a tsv file
the output is {term_id_1: (doc_id, freq), term_id_2: (doc_id, freq), term_id_n:(doc_id, freq) }
where:
term_id_i: is the analyzed term mapped to our vocab
doc_id: is the id of the document processed by the mapper
freq: is the frequency of term in the doc 


"""
# input comes from STDIN (standard input)
word_count = {}
vocab = json.load(fp=open('/Users/ince/Desktop/uni/adm/hws/hw3/vocab.json', 'r', encoding='utf-8'))
for line in sys.stdin:
    filename = os.environ["map_input_file"]
    filename = filename.split('/')[-1].split('.')[0]
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split('\t')[4].split()
    # increase counters
    for word in words:
        word=word.lower();
        print('%s\t%s' % (vocab[word], filename))
       