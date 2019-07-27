import numpy as np
import matplotlib.pyplot as plt


def eig_decomposition(A):
    e_vals, e_vecs = np.linalg.eig(A)
    diag = np.diag(e_vals)
    return e_vecs, diag, np.transpose(e_vecs)

def load_data_rect(x_range, y_range, num):
    x = [x_range[0] + float(i) * (x_range[1] - x_range[0]) / num for i in range(num)]
    y = [y_range[0] + float(i) * (y_range[1] - y_range[0]) / num for i in range(num)]
    d1 = list(zip([x_range[0]] * num, y))
    d2 = list(zip([x_range[1]] * num, y))
    d3 = list(zip(x, [y_range[0]] * num))
    d4 = list(zip(x, [y_range[1]] * num))
    d = d1 + d2 + d3 + d4
    dot = np.array(d)
    return dot

def plot_data(x, format):
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    for x_, f in zip(x, format):
        plt.plot(x_[:, 0], x_[:, 1], f, ms=3)
    plt.show()

def linear_transform(A, x):
    x1 = np.dot(A, np.transpose(x, [1, 0]))
    return np.transpose(x1)

if __name__ == "__main__":
    x = load_data_rect([0, 1], [0, 1], 100)
    A = np.array([[1, 3], [3, 1]])
    u, diag, u_t = eig_decomposition(A)
    print(diag)
    print(u)
    x1 = linear_transform(u_t, x)
    x2 = linear_transform(diag, x1)
    x_m = linear_transform(u, x2)
    x_d = linear_transform(diag, x)

    plot_data([x, x1, x2, x_m], ["ro", "bo", "go", "yo"])
    #plot_data([x, x1, x2, x_m_1], ["ro", "bo", "go", "co"])