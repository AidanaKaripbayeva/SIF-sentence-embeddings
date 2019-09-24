# A-Critique-of-the-Smooth-Inverse-Frequency-Sentence-Embeddings

A Python implementation of the empirical part of the paper:\
**"A Critique of the Smooth Inverse Frequency Sentence Embeddings”** \
*Aidana Karipbayeva, Alena Sorokina, Zhenisbek Assylbekov*.\

# Contacts
**Authors**: Aidana Karipbayeva, Alena Sorokina\
**Pull requests and issues**: aidana.karipbayeva@nu.edu.kz; alena.sorokina@nu.edu.kz 

# Contents
We critically review the smooth inverse frequency sentenceembedding method of Arora, Liang, and Ma (2017), and showinconsistencies in its setup, derivation and evaluation.

**Keywords**: natural language processing, sentence embeddings, smooth inverse frequency

We evaluate and compare 4 sentence embeddings models:
1.	Smooth Inverse Frequency without Principal Component Remove
2.	Smooth Inverse Frequency with Principal Component Remove
3.	Average without Principal Component Remove
4.	Average with Principal Component Remove) 

We performed such comparison on datasets from the SemEval Semantic Textual Similarity (STS) tasks with GLOVE and Word2Vec word embeddings:

a.	Word vectors were trained on the same dataset (Enwik 9), with the same set up (min count = 50, dimension of the word vector = 200). 



- The first file **PMI** is used to obtain the PMI (Pointwise Mutual Information) Matrix. As input we used text8 dataset  (http://mattmahoney.net/dc/text8.zip), and as output we get two files:

	SPPMI matrix in <a href="https://www.codecogs.com/eqnedit.php?latex=R^{V\times&space;V}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R^{V\times&space;V}" title="R^{V\times V}" /></a> , where V is a vocabulary size. (**your_pmi_name.npz**)\
	Vector in <a href="https://www.codecogs.com/eqnedit.php?latex=R^{1\times&space;V}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R^{1\times&space;V}" title="R^{1\times V}" /></a> , which contain the words as strings (**your_column_name**)

- Then, we factorize SSPMI matrix, using different low-rank approximation methods, namely **SVD, QR, NMF**. We suggest to use as word embeddings:\
	for **SVD** - <a href="https://www.codecogs.com/eqnedit.php?latex=U_{d}\Sigma_{d}^{1/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U_{d}\Sigma_{d}^{1/2}" title="U_{d}\Sigma_{d}^{1/2}" /></a> \
	for **NMF** - <a href="https://www.codecogs.com/eqnedit.php?latex=W_{d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{d}" title="W_{d}" /></a> \
	for **QR** – <a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;(&space;RP^{T}&space;\right&space;)_{d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;(&space;RP^{T}&space;\right&space;)_{d}" title="\left ( RP^{T} \right )_{d}" /></a> \
	It should be noted that we are using QR factorization with pivoting and the maximum shape of the PMI matrix that could be factorized will be around 50 000 x 50 000. If more, it can give a memory error.


- Obtained word embeddings (for each method) should be merged with the **your_column_name** file in the terminal. The instructions about this procedure are included in the each file for matrix factorization (**SVD, QR, NMF**).

- We evaluate the performance of the word embeddings using linguistic tasks: analogy and similarity. The **Gensim** library was used, and the code for evaluation is included in each file.
