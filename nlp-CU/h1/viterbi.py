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
                # 这里将bigram和trigram都统计了，实际上下面没有用到bi，只用到tri，所以bi可以不统计
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

    # 求pi_dict, bp_dict
	for k in range(2, len(word_list)):
		u_set = tag_set
		v_set = tag_set
		w_set = tag_set
		if k == 2:
			u_set = ('*', )
			w_set = ('*', )
		elif k == 3:
			w_set = ('*', )
        #遍历不同的u，v，找到最合适的w
		for u, v in itertools.product(u_set, v_set):
			e = efunc(word_list[k], v)
            # candi_list [pi, w]
			candi_list = [((pi_dict[k - 1, w, u] * qfunc(v, w, u) * e), w) for w in w_set]
            # 该处uv情况下最大概率pi，bp为此时的w
			pi, bp = max(candi_list, key = lambda x: x[0])
			pi_dict[k, u, v] = pi
			bp_dict[k, u, v] = bp

    # 最后一个因为是STOP，和前面有一点不一样
	uv_list= [(pi_dict[len(word_list) - 1, u, v] * qfunc('STOP', u, v), (u, v)) \
			for (u, v) in itertools.product(tag_set, tag_set)]
    # 最后找到最大的概率就不用保存了，因为不用再往下求了，此时概率已经是maxp((x1,x2,...,xn, y1,y2,...,yn))
	tagn_1, tagn = max(uv_list, key=lambda x:x[0])[1]
	tag_list = [0] * len(word_list)
	tag_list[-2] = tagn_1
	tag_list[-1] = tagn
    # 反过来根据uv，根据最大概率pi向前递推求w，直到2,这里是tag_list注意，要比word_list长度大2
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
