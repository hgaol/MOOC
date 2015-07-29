# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
Created on Wed Jul 29 16:03:06 2015

@author: hgaolbb
"""

import sys
from collections import defaultdict
import itertools

# [word1, word2, ...]
word_set = set()
tag_set = ('O', 'I-GENE')
# [tag, nums]
tag_count_dict = defaultdict(int)
# [(word, tag), nums]
pair_count_dict = defaultdict(int)

def load_counts(fname):
	"""
	"""
	cfile = file(fname, 'r')
	for l in cfile:
		line = l.strip().split(' ')
		if line:
			if line[1] == 'WORDTAG':
				word_set.add(line[3])
				pair_count_dict[(line[3], line[2])] = int(line[0])
			#elif line[1] == '1-GRAM':
			#	tag_set.add(line[2])
			else:
				tag_count_dict[tuple(line[2:])] = int(line[0])
	return;

def efunc(x, y):
	"""
	e(x|y) = p(x,y) / p(y)
	"""
	if x in word_set:
		return float(pair_count_dict[(x, y)]) / float(tag_count_dict[(y,)])
	else:
		return float(pair_count_dict[('_RARE_', y)]) / float(tag_count_dict[(y,)])

def qfunc(v, w, u):
	"""
	q(v|w,u)
	"""
	return tag_count_dict[w, u, v] / float(tag_count_dict[w, u])	
	
def viterbi(word_list):
	'''
	返回tag序列
	'''
	word_list = ['*', '*'] + word_list
	pi_dict = {(1, '*', '*'): 1}
	bp_dict = {}
	
	for k in range(2, len(word_list)):
		u_set = tag_set
		v_set = tag_set
		w_set = tag_set
		if k == 2:
			u_set = ('*', )
			w_set = ('*', )
		elif k == 3:
			w_set = ('*', )
		for u, v in itertools.product(u_set, v_set):
			e = efunc(word_list[k], v)
			candi_list = [((pi_dict[k - 1, w, u] * qfunc(v, w, u) * e), w) for w in w_set]
			pi, bp = max(candi_list, key = lambda x: x[0])
			pi_dict[k, u, v] = pi
			bp_dict[k, u, v] = bp

	uv_list= [(pi_dict[len(word_list) - 1, u, v] * qfunc('STOP', u, v), (u, v)) \
			for (u, v) in itertools.product(tag_set, tag_set)]
	tagn_1, tagn = max(uv_list, key=lambda x:x[0])[1]
	tag_list = [0] * len(word_list)
	tag_list[-2] = tagn_1
	tag_list[-1] = tagn
	for i in reversed(range(len(tag_list) - 2)):
		tag_list[i] = bp_dict[i + 2, tag_list[i + 1], tag_list[i + 2]]
	return tag_list[2:]

if __name__ == '__main__':
	count_fname = sys.argv[1]
	input_fname = sys.argv[2]
	load_counts(count_fname)
	fout = open('gene_test.p3.out', 'w')
	word_list = []
	cnt = 0
	for line in open(input_fname):
		line = line.strip()
		if not line:
			tag_list = viterbi(word_list)
			for word, tag in zip(word_list, tag_list):
				fout.write('%s %s\n' % (word, tag))
			fout.write('\n')
			word_list = []
		else:
			word_list.append(line)
	
	fout.close()