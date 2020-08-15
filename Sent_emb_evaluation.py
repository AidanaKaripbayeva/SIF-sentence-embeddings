import numpy as np
import multiprocessing as mp
import random,copy,string
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
from collections import Counter
from numpy.linalg import norm
from scipy.stats import pearsonr
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import math

#Uploading the word vectors
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/aidana/glove.840B.300d.txt', binary=False)
model = Word2Vec.load('/home/aidana/Word2Vec_enwik_200d') #load saved model
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/alena/glove/glove_enwik_300.txt', binary=False)

weightfile = '/home/alena/SIF/auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
word2weight = {}
a = 0.001
#Reading the weightfile file
with open(weightfile) as f:
    wlines = f.readlines()
N = 0
for i in wlines:
    i=i.strip()
    if(len(i) > 0):
        i=i.split()
        if(len(i) == 2):
            word2weight[i[0]] = float(i[1])
            N += float(i[1])
        else:
            print(i)

for key, value in word2weight.items(): #creating dictionary with SIF weight for each word
    word2weight[key] = a / (a + value/N)


dim = 200

def compute_pc(X,npc=1):

    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

#Principal Component Removal (PCR)
def remove_pc(X, npc=1):

    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

#SIF without PCR
def sif_no_rem(sent):
    a = 0.001
    em = np.zeros((len(sent), dim))
    for count, sentence in enumerate(sent):
        for word in sentence:
            try:
                w = model[word]
                em[count] = em[count] + word2weight[word] * w
            except:
                w = np.zeros(dim)
        em[count] = em[count]/len(sentence)


    cos = []
    for i in range(len(s1)):
        cos.append(np.dot(em[i], em[i + len(s1)])/ (norm(em[i])*norm(em[i+len(s1)])))

    return pearsonr(labl, cos)


#SIF with PCR
def sif_with_rem(sent):
    a = 0.001
    em = np.zeros((len(sent), dim))
    for count, sentence in enumerate(sent):
        for word in sentence:
            try:
                w = model[word]
                em[count] = em[count] + word2weight[word] * w
            except:
                w = np.zeros(dim)
        em[count] = em[count]/len(sentence)


    rmpc = 1
    emb = remove_pc(em, rmpc)
    em = emb

    cos = []
    for i in range(len(s1)):
        cos.append(np.dot(em[i], em[i + len(s1)])/ (norm(em[i])*norm(em[i+len(s1)])))

    return pearsonr(labl, cos)

#Average without PCR
def ave_no_rem(sent):
    em = np.zeros((len(sent), dim))
    for count, sentence in enumerate(sent):
        for word in sentence:
            try:
                w = model[word]
            except:
                w = np.zeros(dim)
            em[count] = em[count] + w
        em[count] = em[count]/len(sentence)


    cos = []
    for i in range(len(s1)):
        cos.append(np.dot(em[i], em[i + len(s1)])/ (norm(em[i])*norm(em[i+len(s1)])))
    #print("cos", cos)
    return pearsonr(labl, cos)

#Average with PCR
def ave_with_rem(sent):
    em = np.zeros((len(sent), dim))
    for count, sentence in enumerate(sent):
        for word in sentence:
            try:
                w = model[word]
            except:
                w = np.zeros(dim)
            em[count] = em[count] + w
        em[count] = em[count]/len(sentence)


    rmpc = 1
    emb = remove_pc(em, rmpc)
    em = emb

    cos = []
    for i in range(len(s1)):
        cos.append(np.dot(em[i], em[i + len(s1)])/ (norm(em[i])*norm(em[i+len(s1)])))

    return pearsonr(labl, cos)

df = pd.DataFrame(columns = ['dataset','SIF without PCR', 'SIF with PCR', 'Average without PCR', 'Average with PCR'])
farr = ["MSRpar2012",
        "MSRvid2012",
        "OnWN2012",
        "SMTeuro2012",
        "SMTnews2012", # 4
        "FNWN2013",
        "OnWN2013",
        # "SMT2013",
        "headline2013", # 8
        "OnWN2014",
        "deft-forum2014", 
        "deft-news2014",
        "headline2014",
        "images2014",
        "tweet-news2014", # 14
        "answer-forum2015",
        "answer-student2015",
        "belief2015",
        "headline2015",
        "images2015" ]



#Reading STS files
for c, dataset in enumerate(farr):
    s0,s1,labl = [],[],[]
    lines=open("/home/alena/SIF/data/"+dataset,'r').readlines()
    for count, line in enumerate(lines):
        s = line.rstrip().split('\t')
        s0.append([word.lower() for word in word_tokenize(s[0]) if word not in string.punctuation])
        s1.append([word.lower() for word in word_tokenize(s[1]) if word not in string.punctuation])
        labl.append(float(s[2]))

    we =s0+s1

    #
    print(len(s0), len(s1))
    sif_no_rem1 = sif_no_rem(we)[0]
    sif_with_rem1 = sif_with_rem(we)[0]
    ave_no_rem1 = ave_no_rem(we)[0]
    ave_with_rem1 = ave_with_rem(we)[0]
    df.loc[c] = [str(dataset), sif_no_rem1, sif_with_rem1, ave_no_rem1, ave_with_rem1]
    print(dataset)

df.to_excel('Results_of_glove200d.xlsx')
