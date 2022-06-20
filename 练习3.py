import numpy as np

data_x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
data_y = [0, 1, 1, 0]


def sigmoid(v):
    output = 1 / (1 + np.exp(-v))
    return output


def SGD(W, data_x, data_y, lr, epoch):
    v = 0
    for n in range(epoch):
        for i in range(4):
            for j in range(3):
                 v += W[j]*data_x[i][j]
            y_hat = sigmoid(v)
            e = data_y[i] - y_hat
            for k in range(3):
                W[k] += lr*e*sigmoid(v)*(1-sigmoid(v))*data_x[i][k]
    return W


print('练习三')
W = np.zeros(3)
W = SGD(W, data_x, data_y, 0.5, 4000)
y_list = []
for i in range(4):
    v = 0
    for j in range(3):
        v += W[j] * data_x[i][j]
    y_hat = sigmoid(v)
    y_list.append(y_hat)
print('y:', y_list, 'weight:', W)
print('无效，因为训练集无解')
