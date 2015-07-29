# coding: utf8
#! /usr/bin/python

__author__="hgaolbb"
__date__="$July 29, 2015"

import sys
from collections import defaultdict
import math

"""
1-gram
"""

"""
1
"""

# @brief:输出字典word_tag, word_dict, uni_tag
# [param_in]: counts_file   输入文件
# [param_out]: word_tag   以(word, tag)为key的字典，记录其出现的次数
# [param_out]: word_dict   word有哪些
# [param_out]: uni_tag  tag有哪些
def read_counts(counts_file, word_tag, word_dict, uni_tag):

    for l in counts_file:
        line = l.strip().split(' ')
        if line[1] == 'WORDTAG':
            word_tag[(line[3], line[2])] = int(line[0])
            word_dict.append(line[3])
        elif line[1] == '1-GRAM':
            uni_tag[(line[2])] = int(line[0])

# @brief: 构建1-gram字典
def word_with_max_tagger(word_tag, word_dict, uni_tag, word_tag_max):
    for word in word_dict:
        max_tag = ''
        max_val = 0.0
        for tag in uni_tag:
            if float(word_tag[(word, tag)]) / float(uni_tag[(tag)]) > max_val:
                max_val = float(word_tag[(word, tag)]) / float(uni_tag[(tag)])
                max_tag = tag

        word_tag_max[(word)] = max_tag

def tag_gene(word_tag_max, out_f, dev_file):
    for l in dev_file:
        line = l.strip()
        if line:
            if line in word_tag_max:
                out_f.write("%s %s\n" % (line, word_tag_max[(line)]))
            else:
                out_f.write("%s %s\n" % (line, word_tag_max[('_RARE_')]))
        else:
            out_f.write("\n")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit(1)
    counts_file = file(sys.argv[1], 'r')
    dev_file = file(sys.argv[2], 'r')
    word_tag = defaultdict(int)
    uni_tag = defaultdict(int)
    word_dict = []

    word_tag_max = defaultdict(int)
    read_counts(counts_file, word_tag, word_dict, uni_tag)
    counts_file.close()
    word_with_max_tagger(word_tag, word_dict, uni_tag, word_tag_max)
    tag_gene(word_tag_max, sys.stdout, dev_file)
    dev_file.close()
