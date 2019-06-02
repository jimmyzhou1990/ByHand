import numpy as np
import matplotlib.pyplot as plt
from svm.svm import SVM

def load_data_2d_linear(w, b, n, data_range, gap):
    x_pos = []
    while len(x_pos) < n:
        x = np.random.rand(2)*data_range
        if np.dot(w, x) + b >= gap:
            #print(x)
            x_pos.append(x)
    y_pos = np.zeros(len(x_pos))+1

    x_neg = []
    while len(x_neg) < n:
        x = np.random.rand(2)*data_range
        if np.dot(w, x) + b <= -gap:
            #print(x)
            x_neg.append(x)
    y_neg = np.zeros(len(x_pos))-1

    x_out = np.array(x_pos+x_neg)
    y_out = np.concatenate((y_pos, y_neg))

    return x_out, y_out

#y = k * x^2 + b
def load_data_2d_quadratic(k, b, n, data_range=[-10, 10], gap=1):
    wid = data_range[1] - data_range[0]
    x_pos = []
    while len(x_pos) < n:
        x, y = np.random.rand(2)*wid + data_range[0]
        if y - k*np.power(x, 2) - b >= gap:
            x_pos.append([x, y])
    y_pos = np.zeros(len(x_pos)) + 1

    x_neg = []
    while len(x_neg) < n:
        x, y = np.random.rand(2)*wid + data_range[0]
        if y - k * np.power(x, 2) - b <= -gap:
            x_neg.append([x, y])
    y_neg = np.zeros(len(x_pos))-1

    x_out = np.array(x_pos + x_neg)
    y_out = np.concatenate((y_pos, y_neg))

    return x_out, y_out

def plot_2d_linear(data, w, b, support):
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(data[:, 0], data[:, 1], 'ro')

    plt.plot(data[support, 0], data[support, 1], 'go')

    line_x = np.array([0, 10])
    plt.plot(line_x, (-b - w[0]*line_x)/w[1])

    #plt.plot(line_x, (-bt - wt[0]*line_x)/wt[1], ":", color='black')
    plt.show()

def plot_2d_quadratic(data, label, w, b, support):
    plt.xlabel("x")
    plt.ylabel("y")
    #print(label)
    for idx, (xi, label) in enumerate(zip(data, label)):
        #print(xi)
        if idx in support:
            continue
        if label > 0:
            plt.plot(xi[0], xi[1], 'ro')
        else:
            plt.plot(xi[0], xi[1], 'gx')

    plt.plot(data[support, 0], data[support, 1], 'bp')

    x0 = np.array([-10+xi*0.5 for xi in range(40)])
    x1 = (-w[0]*x0*x0-b)/w[1]
    plt.plot(x0, x1)

    plt.show()

if __name__ == '__main__':
    # linear
    # wt = np.array([2, -3])
    # bt = 0.5
    # num = 100
    # rang = 10
    # gap = 3
    # x, y = load_data_2d_linear(wt, bt, num, rang, gap)
    #
    # svm = SVM(x, y, 10000)
    # svm.train()
    #
    # #print(svm.alpha)
    # print(svm.b)
    # print(svm.w)
    # print(svm.supportIndex)
    # print(svm.step_count)
    # supportIndex = list(svm.supportIndex)
    #
    # plot_2d_linear(x, svm.w, svm.b, supportIndex)

    #no-linear
    x, y = load_data_2d_quadratic(0.1, -3, 100, [-10, 10], 0.5)

    svm = SVM(x, y, 10000, 'quadratic')
    svm.train()
    supportIndex = list(svm.supportIndex)

    plot_2d_quadratic(x, y, svm.w, svm.b, supportIndex)



