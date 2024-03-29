# coding: utf8
#! /usr/bin/python

import sys
from collections import defaultdict
import math

"""
将语料库(gene.train)中出现次数少于5的替换为_RARE_
方法：扫描两遍gene.train，第一遍统计每个word个数，存入dict
      然后遍历dict，将number<5的存入rare_set
      第二遍替换，如果该word在rare_set中，则替换为_RARE_
"""

def get_rare_words(corpus_file):
    """
    Read the corpus_file and return a list of rare words
    """
    word_counts = defaultdict(int)
    rare_words = []

    for l in corpus_file:
        line = l.strip()
        if line:
            linew = line.split(' ')
            if (linew[0]) in word_counts:
                word_counts[(linew[0])] += 1
            else:
                word_counts[(linew[0])] = 1

    for key in word_counts:
        if word_counts[key] < 5:
            rare_words.append(key)
    return rare_words

def replace_with_rare(corpus_file, output_file, rare_words):
    """
    Read the corpus_file and replace rare words with '_RARE_'
    """
    for l in corpus_file:
        line = l.strip()
        if line:
            linew = line.split(' ')
            if (linew[0]) in rare_words:
                output_file.write("_RARE_ %s\n" % (linew[1]))
            else:
                output_file.write(line + "\n")
        else:
            output_file.write("\n")

def usage():
    print """
    python ./set_rare.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce new training file.
    """

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = file(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    rare_words = get_rare_words(input)
    input.close()

    try:
        input = file(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    replace_with_rare(input, sys.stdout, rare_words)
