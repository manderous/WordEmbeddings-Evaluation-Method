import numpy as np
from sklearn.model_selection import KFold


def main():
    # 加载数据

    vocab_file = '../data/vocab/SST_dict.txt'  # 词典路径
    vocab_file_low = '../data/vocab/SST_dict_low10.txt'  # 低频词典路径

    with open(vocab_file, 'r', encoding='UTF-8') as f:
        words = []
        freqs = []
        for x in f.readlines():
            temp = x.rstrip().split('\t')
            if int(temp[1]) < 10:
                words.append(temp[0])
                freqs.append(temp[1])

    with open(vocab_file_low, 'w', encoding='UTF-8') as f:
        for i, word in enumerate(words):
            f.write(word+'\t'+freqs[i]+'\n')


if __name__ == "__main__":
    main()