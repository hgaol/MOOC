#!/usr/bin/env python
#encoding:utf8
import sys
import itertools

def generate_and_or_file(fname1, fname2):
    '''
    计算两个alignment文件的交和并，分别写出到文件。
    每个文件必须是如下格式：
    句子编号 外文index 英文index
    所以对于反过来的翻译结果需要用awk互换一下后面两个column的位置。
    此外同一个句子的不同记录必须相邻，而且句子编号从小到大排列。

    其中交文件：and_alignments.txt可作为子作业3的答案提交，
    但是其实应该作为扩展的出发点，进行一些扩展，已达到更好的效果。
    '''
    with open(fname1) as f:
        alignments1 = [tuple(map(int, line.strip().split(' '))) for line in f]
    groups1 = itertools.groupby(alignments1, key=lambda x:x[0])
    dict1 = dict([(key, list(g)) for key, g in groups1])

    with open(fname2) as f:
        alignments2 = [tuple(map(int, line.strip().split(' '))) for line in f]
    groups2 = itertools.groupby(alignments2, key=lambda x:x[0])
    dict2 = dict([(key, list(g)) for key, g in groups2])

    all_keys = set(dict1.keys()) | set(dict2.keys())
    and_dict = {}
    or_dict = {}
    for key in all_keys:
        aligns1 = set(dict1[key])
        aligns2 = set(dict2[key])
        and_dict[key] = aligns1 & aligns2
        or_dict[key] = aligns1 | aligns2

    and_out = open('and_alignments.txt', 'w')
    for key in sorted(and_dict.keys()):
        for item in sorted(and_dict[key], key = lambda x:(x[2], x[1])):
            and_out.write('%s %s %s\n' % item)
    and_out.close()

    or_out = open('or_aligmnents.txt', 'w')
    for key in sorted(or_dict.keys()):
        for item in sorted(or_dict[key], key = lambda x:(x[2], x[1])):
            or_out.write('%s %s %s\n' % item)
    or_out.close()
    return

if __name__ == '__main__':
    generate_and_or_file(sys.argv[1], sys.argv[2])
