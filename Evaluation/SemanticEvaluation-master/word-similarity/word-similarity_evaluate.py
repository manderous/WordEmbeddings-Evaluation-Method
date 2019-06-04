import argparse
import numpy as np
import scipy
from scipy.spatial.distance import pdist
import pandas as pd
import math
import csv

def main():
    model_vectors_filenames = ['scnu_senti_50d_1.txt', 'scnu_senti_50d_2.txt', 'scnu_senti_50d_3.txt', 'scnu_senti_50d_4.txt',
                         'scnu_senti_50d_5.txt', 'scnu_senti_50d_6.txt', 'scnu_senti_50d_7.txt', 'scnu_senti_50d_8.txt',
                         'scnu_senti_50d_9.txt', 'scnu_senti_50d_10.txt', 'scnu_senti_50d_11.txt', 'scnu_senti_50d_12.txt',
                         'scnu_senti_50d_13.txt', 'scnu_senti_50d_14.txt', 'scnu_senti_50d_15.txt', 'scnu_senti_50d_16.txt',
                         'scnu_senti_50d_17.txt', 'scnu_senti_50d_18.txt', 'scnu_senti_50d_19.txt', 'scnu_senti_50d_20.txt',
                         'scnu_senti_50d_21.txt', 'scnu_senti_50d_22.txt', 'scnu_senti_50d_23.txt', 'scnu_senti_50d_24.txt',
                         'scnu_senti_50d_25.txt', 'scnu_senti_50d_26.txt', 'scnu_senti_50d_27.txt', 'scnu_senti_50d_28.txt',
                         'scnu_senti_50d_29.txt', 'scnu_senti_50d_30.txt', 'scnu_senti_50d_31.txt', 'scnu_senti_50d_32.txt',
                         'scnu_senti_50d_33.txt', 'scnu_senti_50d_34.txt']
    model_one_vectors_filenames = ['model_one_t_1.txt', 'model_one_t_2.txt', 'model_one_t_3.txt','model_one_t_4.txt',
                               'model_one_t_5.txt', 'model_one_t_6.txt', 'model_one_t_7.txt']
    base_vectors_filenames = ['cbow_50d.txt']
    model_prefix = '../data/SST_WordEmbedding/model_two/'
    model_one_prefix = '../data/SST_WordEmbedding/model_one/'
    base_prefix = '../data/SST_WordEmbedding/'
    vocab_file = '../data/SST_dict.txt'
    accuracy_flie = 'accuracy.csv'
    accuracy_list = []
    out = open(accuracy_flie, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for j in range(len(model_one_vectors_filenames)):
        vectors_file = model_one_prefix + model_one_vectors_filenames[j]
        file_accuracy = run_file(vocab_file, vectors_file)
        accuracy_list.append(file_accuracy)
        csv_write.writerow(file_accuracy)
    accuracy_arr = np.array(accuracy_list)
    print(accuracy_arr)
    print('write over')

def run_file(vocab_file, vectors_file):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    # parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    # args = parser.parse_args()


    with open(vocab_file, 'r', encoding='UTF-8') as f:
        words = [x.rstrip().split('\t')[0] for x in f.readlines()]
    with open(vectors_file, 'r', encoding='UTF-8') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '</s>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return evaluate_vectors(W_norm, vocab, ivocab)

def evaluate_vectors(W, vocab, ivocab):
    """Evaluate the trained word vectors on a variety of tasks"""
    filenames = ['EN-MC-30.txt', 'EN-MEN-TR-3k.txt', 'EN-MTurk-287.txt', 'EN-MTurk-771.txt',
                 'EN-RG-65.txt', 'EN-RW-STANFORD.txt', 'EN-SIMLEX-999.txt', 'EN-SimVerb-3500.txt',
                 'EN-VERB-143.txt', 'EN-WS-353-ALL.txt', 'EN-WS-353-REL.txt', 'EN-WS-353-SIM.txt',
                 'EN-YP-130.txt', 'SCWS.txt']
    prefix = './similarity_corpus/'

    count_all = 0 # count all the pairs
    accuracy_all = 0 # count all questions' correct
    accuracy = 0
    full_count = 0  # count all questions, including those with unknown words

    accuracy_collect = []  # 收集所有的值

    for i in range(len(filenames)):
        path = prefix + filenames[i]
        with open(path, 'r') as f:
            full_data = [line.rstrip().replace('\t', ' ').split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x[0:2])]

        indices = np.array([[vocab[word] for word in row[0:2]] for row in data])
        score = np.array([float(row[2]) for row in data]).T
        ind1, ind2 = indices.T

        print("%s:" % filenames[i])
        num_iter = len(ind1)
        dist = []
        for j in range(num_iter):
            v1 = W[ind1[j]]
            v2 = W[ind2[j]]
            dist.append(1 - math.fabs(pdist(np.vstack([v1,v2]), 'cosine')))
        df = pd.DataFrame({'dist':dist, 'score':score})
        accuracy = df.corr().loc['dist','score']
        accuracy_all += accuracy
        print('word pairs number:',str(num_iter), 'accuracy:', str(accuracy*100),'%')
        accuracy_collect.append(accuracy*100)
    print('total accuracy:', str(accuracy_all*10))
    accuracy_collect.append(accuracy_all*10)
    return accuracy_collect


if __name__ == "__main__":
    main()
