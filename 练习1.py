import numpy as np

data_x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
data_y = [0, 0, 1, 1]


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


def Batch(W, data_x, data_y, lr, epoch):
    v = 0
    for n in range(epoch):
        w_list = np.zeros((4, 3))
        for i in range(4):
            for j in range(3):
                v += W[j] * data_x[i][j]
            y_hat = sigmoid(v)
            e = data_y[i] - y_hat
            for k in range(3):
                w_list[i][k] = (lr * e * sigmoid(v) * (1 - sigmoid(v)) * data_x[i][k])
        W += (w_list[0]+w_list[1]+w_list[2]+w_list[3])/4
    return W


def MiniBatch(W, data_x, data_y, lr, epoch, batch_size):
    v = 0
    for n in range(epoch):
        w_list = np.zeros((2, 3))
        for i in range(batch_size):
            for j in range(3):
                v += W[j] * data_x[i][j]
            y_hat = sigmoid(v)
            e = data_y[i] - y_hat
            for k in range(3):
                w_list[i][k] = (lr * e * sigmoid(v) * (1 - sigmoid(v)) * data_x[i][k])
        W += (w_list[0] + w_list[1]) / 2
        for i in range(batch_size):
            for j in range(3):
                v += W[j] * data_x[i+batch_size][j]
            y_hat = sigmoid(v)
            e = data_y[i+batch_size] - y_hat
            for k in range(3):
                w_list[i][k] = (lr * e * sigmoid(v) * (1 - sigmoid(v)) * data_x[i+batch_size][k])
        W += (w_list[0] + w_list[1]) / 2
    return W


print("练习一")
W = np.zeros((3, 3))
W[0] = SGD(W[0], data_x, data_y, 0.8, 1000)
W[1] = Batch(W[1], data_x, data_y, 0.8, 4000)
W[2] = MiniBatch(W[2], data_x, data_y, 0.8, 2000, 2)
for n in range(3):
    y_list = []
    for i in range(4):
        v = 0
        for j in range(3):
            v += W[n][j] * data_x[i][j]
        y_hat = sigmoid(v)
        y_list.append(y_hat)
    print('y:', y_list, 'weight:', W[n])

