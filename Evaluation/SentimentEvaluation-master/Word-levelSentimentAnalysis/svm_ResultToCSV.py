import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
import csv



def clip(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def default_ker(x, z):
    return x.dot(z.T)


def svm_smo(x, y, ker, C, max_iter, epsilon=1e-5):
    # initialization
    n, _ = x.shape
    alpha = np.zeros((n,))

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = ker(x[i], x[j])

    iter = 0
    while iter <= max_iter:
        if iter%100 == 0:
            print('smo算法第',iter,'轮') # 每100轮显示输出一次

        for i in range(n):
            # randomly choose an index j, where j is not equal to i
            j = np.random.randint(low=0, high=n - 1)
            j += (j >= i)

            # update alpha_i
            eta = K[j, j] + K[i, i] - 2.0 * K[i, j]
            if np.abs(eta) < epsilon:
                continue

            e_i = (K[:, i] * alpha * y).sum() - y[i]
            e_j = (K[:, j] * alpha * y).sum() - y[j]
            alpha_i = alpha[i] - y[i] * (e_i - e_j) / eta

            # clip alpha_i
            lower, upper = 0, C
            zeta = alpha[i] * y[i] + alpha[j] * y[j]
            if y[i] == y[j]:
                lower = max(lower, zeta / y[j] - C)
                upper = min(upper, zeta / y[j])
            else:
                lower = max(lower, -zeta / y[j])
                upper = min(upper, C - zeta / y[j])

            alpha_i = clip(alpha_i, lower, upper)

            # update alpha_j
            alpha_j = (zeta - y[i] * alpha_i) / y[j]
            if np.abs(alpha_i - alpha[i]) > epsilon or np.abs(alpha_j - alpha[j]) > epsilon:
                alpha[i] = alpha_i
                alpha[j] = alpha_j

        iter += 1

    # calculate b
    b = 0
    for i in range(n):
        if epsilon < alpha[i] < C - epsilon:
            b = y[i] - (y * alpha * K[:, i]).sum()

    def f(X):  # predict the point X based on alpha and b
        results = []
        for k in range(X.shape[0]):
            result = b
            for i in range(n):
                result += y[i] * alpha[i] * ker(x[i], X[k])
            results.append(result)
        return np.array(results)

    return f, alpha, b


def data_visualization(x, y):
    import matplotlib.pyplot as plt
    category = {'+1': [], '-1': []}
    for point, label in zip(x, y):
        if label == 1.0:
            category['+1'].append(point)
        else:
            category['-1'].append(point)
    ax = plt.subplot(111, projection='3d')
    for label, pts in category.items():
        pts = np.array(pts)
        # 取词向量中的前三维显示，颜色不同区分label
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=label)


    plt.show()

# 将svm预测结果转为-1和1的标签
def prediction_to_label(prediction):
    _prediction = []
    for train_p in prediction:
        if train_p > 0:
            _prediction.append(1)
        else:
            _prediction.append(-1)
    return np.array(_prediction)


def run_file(vocab_file, vectors_file, senti_vocab_file, normalize_flag, fold_splits):
    # 加载数据
    with open(vocab_file, 'r', encoding='UTF-8') as f:
        words = [x.rstrip().split('\t')[0] for x in f.readlines()]
    with open(vectors_file, 'r', encoding='UTF-8') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0]=='':
                continue
            if vals[0] in words:
                vectors[vals[0]] = [float(x) for x in vals[1:]]
    with open(senti_vocab_file, 'r', encoding='UTF-8') as f:
        senti_vacab = {}
        senti_words = []
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[1] == 'positive':
                senti_vacab[vals[0]] = 1
                if vals[0] not in senti_words:
                    senti_words.append(vals[0])
            elif vals[1] == 'negative':
                senti_vacab[vals[0]] = -1
                if vals[0] not in senti_words:
                    senti_words.append(vals[0])

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T


    new_words = [new for new in words if new in senti_words] # words与senti_words取交集
    new_words_size = len(new_words)
    senti_words_size = len(senti_words)
    print('覆盖情感词典百分比：',new_words_size/senti_words_size*100,'%')
    print("\n\n")
    x = np.zeros((new_words_size, vector_dim))  # word
    y = [] # label
    for idx, word in enumerate(new_words):
        if normalize_flag == True:
            x[idx, :] = W_norm[vocab[word]]
        else:
            x[idx, :] = W[vocab[word]]
        y.append(senti_vacab[word])
    y = np.array(y)


    # 三维数据可视化
    # data_visualization(x, y)


    # 交叉验证
    train_acc = []
    test_acc = []
    kf = KFold(n_splits=fold_splits)
    count = 1
    for train_idx, test_idx in kf.split(range(new_words_size)):
        x_train = x[train_idx, :]
        y_train = y[train_idx]
        x_test = x[test_idx, :]
        y_test = y[test_idx]
        print('第', count, '轮交叉验证')
        # 运行svm分类器
        model, alpha, bias = svm_smo(x_train, y_train, default_ker, 1e2, 1000)
        # 每一轮的评估
        train_prediction = model(x_train)  # 训练集结果
        test_prediction = model(x_test)  # 测试集结果
        _train_prediction = prediction_to_label(train_prediction)
        _test_prediction = prediction_to_label(test_prediction)
        _train_acc = np.sum(_train_prediction == y_train) / len(y_train)
        _test_acc = np.sum(_test_prediction == y_test) / len(y_test)
        train_acc.append(_train_acc)
        test_acc.append(_test_acc)
        print('第',count,'轮交叉验证的训练集准确率为：',_train_acc,'测试集准确率为：',_test_acc)
        count += 1

    # 总体评估
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    train_acc_average = np.sum(train_acc) / fold_splits
    test_acc_average = np.sum(test_acc) / fold_splits

    print('\n',fold_splits,'折交叉验证的平均训练集准确率为：',train_acc_average,'平均测试集准确率为：',test_acc_average)
    return train_acc_average, test_acc_average


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
    base_vectors_filenames = ['SSWE/cw_50d.txt',
                              'glove_50d.txt', 'SSWE/sswe-h_50d.txt', 'SSWE/sswe-r_50d.txt', 'SSWE/sswe-u_50d.txt',
                              'Yang/DLJT1.txt', 'Yang/DLJC1.txt', 'Yang/DLJT2.txt', 'Yang/DLJC2.txt',
                              'model_one/scun_senti_glove_t.txt', 'model_one/scun_senti_glove_c.txt']
    model_one_vectors_filenames = ['model_one_t_1.txt', 'model_one_t_2.txt', 'model_one_t_3.txt', 'model_one_t_4.txt',
                                   'model_one_t_5.txt', 'model_one_t_6.txt', 'model_one_t_7.txt']
    cbow_filenames = ['cbow_50d.txt']
    model_prefix = '../data/SST/model_two/'
    base_prefix = '../data/SST/'
    model_one_prefix = '../data/SST/model_one/'

    vocab_file = '../data/vocab/SST_dict.txt'  # 词典路径
    senti_vocab_file = '../data/senti_vocab/E_EMNLP.txt'  # 情感词典路径
    normalize_flag = True  # 是否对词向量进行归一化
    fold_splits = 10  # 交叉验证折数

    train_test_acc_average_list = []
    accuracy_flie = 'accuracy_base_MPQA.csv'
    out = open(accuracy_flie, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for j in range(len(model_one_vectors_filenames)):
        vectors_file = model_one_prefix + model_one_vectors_filenames[j] # 词向量路径
        train_acc_average, test_acc_average = run_file(vocab_file, vectors_file, senti_vocab_file, normalize_flag, fold_splits)
        temp = [train_acc_average, test_acc_average]
        train_test_acc_average_list.append(temp)
        csv_write.writerow(temp)
        train_test_acc_average_arr = np.array(train_test_acc_average_list)
    print(train_test_acc_average_arr)
    print('write over')


if __name__ == "__main__":
    main()