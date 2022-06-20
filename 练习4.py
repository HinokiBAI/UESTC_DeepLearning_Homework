import numpy as np

data_x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
data_y = [0, 1, 1, 0]


def sigmoid(v):
    output = 1 / (1 + np.exp(-v))
    return output


def NN(W1, W2, data_x, data_y, lr, epoch):
    v = 0
    for n in range(epoch):
        for i in range(4):
            v1 = np.dot(W1, data_x[i].transpose())
            y1 = sigmoid(v1)
            v2 = np.dot(W2, y1)
            y_hat = sigmoid(v2)
            e = data_y[i] - y_hat
            e1 = W2.transpose()*y_hat*(1-y_hat)*e
            y1 = y1.reshape(4, 1)
            delta1 = y1*(1-y1)*e1
            dW1 = np.dot(lr*delta1, data_x[i].reshape(1, 3))
            W1 = W1+dW1
            delta2 = y_hat*(1-y_hat)*e
            W2 = W2+lr*y_hat*(1-y_hat)*e*y1.transpose()
    return W1, W2


print('logistic回归')
W1 = np.random.randn(4, 3)
W2 = np.random.randn(1, 4)
W1, W2 = NN(W1, W2, data_x, data_y, 0.9, 10000)
y_list = []
for i in range(4):
    v1 = np.dot(W1, data_x[i].transpose())
    y1 = sigmoid(v1)
    v = np.dot(W2, y1)
    y_hat = sigmoid(v)
    y_list.append(y_hat)
print('y:', y_list)
