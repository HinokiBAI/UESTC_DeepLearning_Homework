import numpy as np


def sigmoid(v):
    output = 1 / (1 + np.exp(-v))
    return output


def softmax(x):
    ex = np.exp(x)
    y = ex/np.sum(ex)
    return y


def multi_classification(W1, W2, data_x, data_y, epoch, lr):
    for n_epoch in range(epoch):
        for i in range(5):
            x = data_x[i].reshape(25, 1)
            y = data_y[i]
            v1 = np.dot(x.transpose(), W1)
            y1 = sigmoid(v1)
            v2 = np.dot(y1, W2)
            y_hat = softmax(v2)
            e = y - y_hat
            # print(e.shape)
            e1 = np.dot(e, W2.transpose())
            delta1 = y1*(1-y1)*e1
            # print(delta1.shape)
            dW1 = np.dot(data_x[i].reshape(len(data_x[i].flatten()), 1), lr * delta1)
            W1 = W1 + dW1
            W2 = W2 + lr * y_hat * (1 - y_hat) * e * y1.transpose()
    return W1, W2


def identify(y_hat):
    # m = max(y_hat)
    m_id = np.where(y_hat == np.max(y_hat))
    number = m_id[1]
    return number+1


print("练习9")
data_x = np.zeros([5, 5, 5])
data_x[0] = [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0]]
data_x[1] = [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
data_x[2] = [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
data_x[3] = [[0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0]]
data_x[4] = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
data_y = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

W1 = np.random.randn(25, 50)
W2 = np.random.randn(50, 5)
W1, W2 = multi_classification(W1, W2, data_x, data_y, epoch=10000, lr=0.9)
for i in range(5):
    x = data_x[i].reshape(25, 1)
    v1 = np.dot(x.transpose(), W1)
    y1 = sigmoid(v1)
    v = np.dot(y1, W2)
    y_hat = softmax(v)
    identified_num = identify(y_hat)
    print('real:', data_y[i], "predict:", identified_num)
