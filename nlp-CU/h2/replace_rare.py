# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:44:17 2015

@author: g
@biref: 统计训练集中每个单词（叶子）的数量，小于5的替换为_RARE_
@usage: python replace_rare.py
@method: 第一次统计单词出现次数，构建字典;
         第二次根据字典，将小于5次的单词替换为_RARE_
"""

import sys, json
from collections import defaultdict

def get_words(tree):
    '''
    递归找到叶子节点(word)，将所有word放在list中返回
    '''
	if len(tree) == 2:
		return tree[1]
	else:
		return get_words(tree[1]) + get_words(tree[2])

def replace_rare_tree(tree, count_dict):
    '''
    递归，到叶子节点(word)时，如果该word小于5次则替换为_RARE_
    '''
	if len(tree) == 2:
		if count_dict[tree[1]] < 5:
			tree[1] = '_RARE_'
		return
	replace_rare_tree(tree[1], count_dict)
	replace_rare_tree(tree[2], count_dict)
	return

def rep_rare_file(fname, out_name):
    count_dict = defaultdict(int)
    with open(fname) as f:
        for line in f:
            tree = json.loads(line.strip())
            words = get_words(tree)
            for w in words:
                count_dict[w] += 1

    fout = open(out_name, 'w')
    with open(fname) as f:
        for line in f:
            tree = json.loads(line.strip())
            replace_rare_tree(tree, count_dict)
            # tree已经焕然一新，write吧
            fout.write('%s\n' % json.dumps(tree))
    fout.close()

if __name__ == '__main__':
    fname = sys.argv[1]
    rep_rare_file(fname, fname + '.rep_rare')
