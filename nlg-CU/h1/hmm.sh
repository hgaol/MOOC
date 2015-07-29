python set_rare.py gene.train > gene.train2
# python count_freqs.py gene.train > gene.counts
# p1
python count_freqs.py gene.train2 > gene.counts2
python gene_tagger_unigram.py gene.counts2 gene.dev > gene_dev.p1.out
python eval_gene_tagger.py gene.key gene_dev.p1.out

#p2
python viterbi.py gene.counts2 gene.dev
python eval_gene_tagger.py gene.key gene_dev.p2.out
