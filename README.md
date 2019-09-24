# A-Critique-of-the-Smooth-Inverse-Frequency-Sentence-Embeddings

A Python implementation of the empirical part of the paper:\
**"A Critique of the Smooth Inverse Frequency Sentence Embeddings”** \
*Aidana Karipbayeva, Alena Sorokina, Zhenisbek Assylbekov.* 

# Contacts
**Authors**: Aidana Karipbayeva, Alena Sorokina\
**Pull requests and issues**: aidana.karipbayeva@nu.edu.kz; alena.sorokina@nu.edu.kz 

# Contents
We critically review the smooth inverse frequency sentence embedding method of Arora, Liang, and Ma (2017), and show inconsistencies in its setup, derivation and evaluation.

**Keywords**: natural language processing, sentence embeddings, smooth inverse frequency

In empirical part of our paper, we evaluate and compare 4 sentence embeddings models:
1.	Smooth Inverse Frequency without Principal Component Remove
2.	Smooth Inverse Frequency with Principal Component Remove
3.	Average without Principal Component Remove
4.	Average with Principal Component Remove) 


We performed such comparison on datasets from the SemEval Semantic Textual Similarity (STS) tasks (http://ixa2.si.ehu.es/stswiki/index.php/Main_Page, test datasets) with GLOVE and Word2Vec word embeddings:

a.	Glove and Word2Vec word vectors were trained on the same dataset (Enwik 9), with the same set up (min count = 50, dimension of the word vector = 200). (Code for the customized training of the word models: /data/Training Word2Vec model with custom set up.ipynb). \

b.	Pre-trained GLOVE word vectors (Common Crawl, 840B tokens, 2.2M vocab, cased, 300d vectors) (can be downloaded from https://nlp.stanford.edu/projects/glove/). \
	Word2Vec vectors trained on Enwik 9, with min_count = 50, window size =2, vector dimension = 300. (Code for the customized training of the word models: /data/Training Word2Vec model with custom set up.ipynb)

Data also contains txt file with words and its frequencies (enwiki_vocab_min200.txt), which is used in SIF model implementation. 
