# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:44:17 2015

@author: g
"""

import sys, json
from collections import defaultdict

def get_words(tree):
	if len(tree) == 2:
		return tree[1]
	else:
		return get_words(tree[1]) + get_words(tree[2])

def replace_rare_tree(tree, count_dict):
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
            fout.write('%s\n' % json.dumps(tree))
    fout.close()

if __name__ == '__main__':
    fname = sys.argv[1]
    rep_rare_file(fname, fname + '.rep_rare')
