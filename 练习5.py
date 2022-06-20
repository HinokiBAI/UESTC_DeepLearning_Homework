import numpy as np

data_x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
data_y = [0, 1, 1, 0]


def sigmoid(v):
    output = 1 / (1 + np.exp(-v))
    return output


def BP(W1, W2, data_x, data_y, lr, epoch):
    v = 0
    for n in range(epoch):
        for i in range(4):
            v1 = np.dot(W1, data_x[i].transpose())
            y1 = sigmoid(v1)
            v2 = np.dot(W2, y1)
            y_hat = sigmoid(v2)
            e = data_y[i] - y_hat
            e1 = W2.transpose()*y_hat*(1-y_hat)*e
            y1 = y1.reshape(len(y1), 1)
            delta1 = y1*(1-y1)*e1
            dW1 = np.dot(lr*delta1, data_x[i].reshape(1, len(data_x[i])))
            W1 = W1+dW1
            delta2 = y_hat*(1-y_hat)*e
            W2 = W2+lr*y_hat*(1-y_hat)*e*y1.transpose()
    return W1, W2


print('练习五')
W1 = np.random.randn(3, 3)
W2 = np.random.randn(1, 3)
W1, W2 = BP(W1, W2, data_x, data_y, 0.5, 10000)
y_list = []
for i in range(4):
    v1 = np.dot(W1, data_x[i].transpose())
    y1 = sigmoid(v1)
    v = np.dot(W2, y1)
    y_hat = sigmoid(v)
    y_list.append(y_hat)
print('节点为3，y:', y_list)

W1 = np.random.randn(5, 3)
W2 = np.random.randn(1, 5)
W1, W2 = BP(W1, W2, data_x, data_y, 0.5, 10000)
y_list = []
for i in range(4):
    v1 = np.dot(W1, data_x[i].transpose())
    y1 = sigmoid(v1)
    v = np.dot(W2, y1)
    y_hat = sigmoid(v)
    y_list.append(y_hat)
print('节点为5，y:', y_list)

W1 = np.random.randn(2, 3)
W2 = np.random.randn(1, 2)
W1, W2 = BP(W1, W2, data_x, data_y, 0.1, 50000)
y_list = []
for i in range(4):
    v1 = np.dot(W1, data_x[i].transpose())
    y1 = sigmoid(v1)
    v = np.dot(W2, y1)
    y_hat = sigmoid(v)
    y_list.append(y_hat)
print('节点为2，y:', y_list)
print('减小学习率，可以使网络更快收敛')
