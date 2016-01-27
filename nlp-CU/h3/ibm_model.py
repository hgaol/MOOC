#!/usr/bin/env python
#encoding:utf8
import itertools
import collections
import cPickle

NULL = '###'

def compute_init_t_params(foreign_sentences, local_sentences):
    '''
    计算迭代的初始t参数。
    '''
    count_dict = collections.defaultdict(list)
    for i in range(len(local_sentences)):
        for word in local_sentences[i]:
            count_dict[word].append(i)
    t_params = {}
    foreign_sentence_sets = [set(sentence) for sentence in foreign_sentences]
    for local_word, occur_list in count_dict.items():
        occur_sentences = (foreign_sentence_sets[i] for i in occur_list)
        occur_word_set = reduce(lambda x, y: x|y, occur_sentences)
        uniform_prob = 1.0 / len(occur_word_set)
        for foreign_word in occur_word_set:
            t_params[foreign_word, local_word] = uniform_prob

    return t_params

def compute_init_q_params(foreign_sentences, local_sentences):
    '''
    计算迭代的初始q参数。
    '''
    q_params = {}
    f_length_list = [len(f_sentence) for f_sentence in foreign_sentences]
    l_length_list = [len(l_sentence) for l_sentence in local_sentences]
    for l, m in zip(l_length_list, f_length_list):
        for i, j in itertools.product(range(m), range(l)):
            # 不用l + 1因为句子已经人为补上了NULL。
            q_params[j, i, l, m] = 1.0 / l

    return q_params

def train_t_q_params_model_2(foreign_fname, local_fname, t_params):
    '''
    用IBM Model2训练t和q参数。
    '''
    with open(foreign_fname) as f:
        foreign_sentences = [line.strip().split(' ') for line in f]
    with open(local_fname) as f:
        # 为每个句子加入NULL前缀。
        local_sentences = [[NULL] + line.strip().split(' ') for line in f]
    print 'Init t_params size :', len(t_params)
    q_params = compute_init_q_params(foreign_sentences, local_sentences)
    train_size = len(foreign_sentences)

    TOTAL_ITER_CNT = 5
    for n in range(TOTAL_ITER_CNT):
        print 'Iteration NO.%s' % (n + 1)
        pair_dict = collections.defaultdict(float)
        word_dict = collections.defaultdict(float)

        ali_dict = collections.defaultdict(float)
        pos_dict = collections.defaultdict(float)

        for k in range(train_size):
            if k % 1000 == 0:
                print '%s sentence pairs processed.' % k
            f_sentence = foreign_sentences[k]
            e_sentence = local_sentences[k]
            m = len(f_sentence)
            l = len(e_sentence)
            for i, j in itertools.product(range(len(f_sentence)), range(len(e_sentence))):
                delta = q_params[(j, i, l, m)] * t_params[(f_sentence[i], e_sentence[j])] / \
                        sum(q_params[(jj, i, l, m)] * t_params[(f_sentence[i], e_sentence[jj])] \
                        for jj in range(l))
                pair_dict[(e_sentence[j], f_sentence[i])] += delta
                word_dict[e_sentence[j]] += delta

                ali_dict[(j, i, l, m)] += delta
                pos_dict[(i, l, m)] += delta


        # 更新t参数。
        for e_word, f_word in pair_dict.keys():
            t_params[(f_word, e_word)] = pair_dict[(e_word, f_word)] / word_dict[e_word]

        # 更新q参数。
        for j, i, l, m in ali_dict.keys():
            q_params[(j, i, l, m)] = ali_dict[(j, i, l, m)] / pos_dict[(i, l, m)]

        print 't_params after %s iterations:' % (n + 1)
        print len(t_params)
        #raw_input('>>>')

    return t_params, q_params

def train_t_params_model_1(foreign_fname, local_fname):
    '''
    根据输入的翻译对应文件，训练t参数。
    '''
    with open(foreign_fname) as f:
        foreign_sentences = [line.strip().split(' ') for line in f]
    with open(local_fname) as f:
        # 为每个句子加入NULL前缀。
        local_sentences = [[NULL] + line.strip().split(' ') for line in f]
    t_params = compute_init_t_params(foreign_sentences, local_sentences)
    print 'Init t_params size :', len(t_params)
    train_size = len(foreign_sentences)

    TOTAL_ITER_CNT = 5
    for n in range(TOTAL_ITER_CNT):
        print 'Iteration NO.%s' % (n + 1)
        pair_dict = collections.defaultdict(float)
        word_dict = collections.defaultdict(float)

        for k in range(train_size):
            if k % 1000 == 0:
                print '%s sentence pairs processed.' % k
            f_sentence = foreign_sentences[k]
            e_sentence = local_sentences[k]
            m = len(f_sentence)
            l = len(e_sentence)
            for i, j in itertools.product(range(len(f_sentence)), range(len(e_sentence))):
                delta = t_params[(f_sentence[i], e_sentence[j])] / sum(t_params[(f_sentence[i], e_sentence[jj])] \
                        for jj in range(l))
                pair_dict[(e_sentence[j], f_sentence[i])] += delta
                word_dict[e_sentence[j]] += delta

        # 更新t参数。
        for e_word, f_word in pair_dict.keys():
            t_params[(f_word, e_word)] = pair_dict[(e_word, f_word)] / word_dict[e_word]

        print 't_params after %s iterations:' % (n + 1)
        print len(t_params)
        #raw_input('>>>')

    return t_params

def compute_alignment_for_pair(f_sentence, e_sentence, t_params, q_params):
    '''
    为一对句子计算alignment。
    '''
    alignment_list = []
    l = len(e_sentence)
    m = len(f_sentence)
    for i, f_word in enumerate(f_sentence):
        candidates = [(j, q_params[(j, i, l, m)] * t_params[(f_word, e_word)]) \
                for j, e_word in enumerate(e_sentence)]

        alignment = max(candidates, key=lambda x:x[1])[0]
        #alignment_list.append((alignment - 1, i))
        if alignment != 0:
            # 因为英文多了一个NULL，所以要下标减一。
            alignment_list.append((alignment - 1, i))

    return alignment_list

def compute_alignment_model(foreign_fname, local_fname, t_params, q_params):
    '''
    根据训练好的t参数，计算语言测试集中的alignment。
    '''
    f_out = open('es_to_en.txt', 'w')
    with open(foreign_fname) as f:
        foreign_sentences = [line.strip().split(' ') for line in f]
    with open(local_fname) as f:
        local_sentences = [[NULL] + line.strip().split(' ') for line in f]
    
    for i, pair in enumerate(zip(foreign_sentences, local_sentences)):
        f_sentence, e_sentence = pair
        alignment_pairs = compute_alignment_for_pair(f_sentence, e_sentence, t_params, q_params)
        for e, f in alignment_pairs:
            f_out.write('%s %s %s\n' % (i + 1, e + 1, f + 1))
    f_out.close()
    return

if __name__ == '__main__':
	
    print 'training t params with ibm model 1...'
    t_params = train_t_params_model_1('corpus.en', 'corpus.es')
    with open('en_to_es_data/t_params_model_1.pickle', 'wb') as f:
        cPickle.dump(t_params, f, cPickle.HIGHEST_PROTOCOL)
    print 'training t params and q params with ibm model 1...'
    t_params, q_params = train_t_q_params_model_2('corpus.en', 'corpus.es', t_params)
    
    '''
    with open('es_to_en_data/t_params_model_2.pickle', 'rb') as f:
        #t_params = cPickle.dump(t_params, f, cPickle.HIGHEST_PROTOCOL)
        t_params = cPickle.load(f)
    with open('es_to_en_data/q_params_model_2.pickle', 'rb') as f:
        q_params = cPickle.load(f)
    print 'decoding alignments...'
    #compute_alignment_model('dev.en', 'dev.es', t_params, q_params)
    compute_alignment_model('test.es', 'test.en', t_params, q_params)
    '''