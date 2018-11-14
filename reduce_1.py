#!/usr/bin/env python



# input comes from STDIN
"""
 w= {}
 s = line.strip().split()
    if s[0] not in w:
        w[s[0]] = s[1]
    else: 
        l = w

w = {}
for line in sys.stdin:
    line = line.strip()
    word, filename = line.split('\t')
    try:
        l = w[word]
        l.append(filename)
        w[word] = l 
    except:
        w[word] = [filename]

print(w)
"""


#!/usr/bin/env python


from itertools import groupby
from operator import itemgetter
import sys

def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.rstrip().split(separator, 1)
w = {}
def main(separator='\t'):
    # input comes from STDIN (standard input)
    data = read_mapper_output(sys.stdin, separator=separator)
    # groupby groups multiple word-count pairs by word,
    # and creates an iterator that returns consecutive keys and their group:
    #   current_word - string containing a word (the key)
    #   group - iterator yielding all ["&lt;current_word&gt;", "&lt;count&gt;"] items
    
    for current_word, group in groupby(data, itemgetter(0)):
        try:
            total_count =  list(count for current_word, count in group)
            w.update({current_word:total_count})
        except ValueError:
            # count was not a number, so silently discard this item
            pass
    print(w)
    
if __name__ == "__main__":
    main()