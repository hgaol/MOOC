#!/usr/bin/env python
#encoding:utf8
import sys
import json
import itertools
import collections

count_dict = {}
word_set = set()

def load_counts(fname):
    with open(fname) as f:
        for line in f:
            tokens = line.strip().split(' ')
            count = int(tokens[0])
            count_type = tokens[1]
            count_dict[tuple(tokens[2:])] = count
            if count_type == 'UNARYRULE':
                word_set.add(tokens[3])
    return

def q_func(rule_left, *rule_right):
    if len(rule_right) == 1 and rule_right[0] not in word_set:
        rule_right = ('_RARE_', )
    top = count_dict[tuple([rule_left] + list(rule_right))]
    down = count_dict[(rule_left,)]
    return float(top) / down

def words_to_tree(words, first, s, last, X, Y, Z, bp_dict):
    if first == last:
        return [X, words[first]]
    ret = [X, [], []]
    ret_tokens = bp_dict[(first, s, Y)]
    if type(ret_tokens) == str:
        ret[1] = [Y, words[first]]
    else:
        Y1, Y2, s1 = ret_tokens
        ret[1] = words_to_tree(words, first, s1, s, Y, Y1, Y2, bp_dict)
    ret_tokens = bp_dict[(s + 1, last, Z)]
    if type(ret_tokens) == str:
        ret[2] = [Z, words[s + 1]]
    else:
        Z1, Z2, s2 = bp_dict[(s + 1, last, Z)]
        ret[2] = words_to_tree(words, s + 1, s2, last, Z, Z1, Z2, bp_dict)
    return ret

def decode_line(line):
    source = line
    pi_dict = collections.defaultdict(float)
    bp_dict = {}
    # Define base cases.
    words = line.split(' ')
    word_index_dict = dict((w, i) for i, w in enumerate(words))
    # print 'word_index_dict', word_index_dict
    for i, w in enumerate(words):
        if w not in word_set:
            w = '_RARE_'
        for count_key in count_dict:
            if len(count_key) != 2:
                continue
            pos, terminal = count_key
            if terminal == w:
                pi_dict[(i, i, pos)] = q_func(pos, terminal)
                bp_dict[(i, i, pos)] = terminal

    # Dynamically compute pi and remember footprint bp.
    for step in range(1, len(words)):
        for i in range(len(words) - step):
            j = i + step
            for count_key in count_dict:
                if len(count_key) != 3:
                    continue
                X, Y, Z = count_key
                for s in range(i, j):
                    current_q = q_func(X, Y, Z) * pi_dict[(i, s, Y)] * pi_dict[(s + 1, j, Z)]
                    if current_q > pi_dict[(i, j, X)]:
                        pi_dict[(i, j, X)] = current_q
                        bp_dict[(i, j, X)] = (Y, Z, s)
    maxY, maxZ, maxs = bp_dict[(0, len(words) - 1, 'SBARQ')]
    word_tree = words_to_tree(source.split(' '), 0, maxs, len(words) - 1, 'SBARQ', maxY, maxZ, bp_dict)

    return word_tree


def decode_file(fname):
    fin = open(fname)
    fout = open(fname + '.out', 'w', 1)
    for line in fin:
        tree = decode_line(line.strip())
        fout.write('%s\n' % json.dumps(tree))
    fin.close()
    fout.close()

if __name__ == '__main__':
    fname = sys.argv[1]
    count_fname = sys.argv[2]
    load_counts(count_fname)
    decode_file(fname)
