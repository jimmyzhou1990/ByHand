import sklearn.datasets as datasets
import numpy as np
from numpy import *
from sklearn.preprocessing import StandardScaler
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# CRIM：城镇人均犯罪率。
# ZN：住宅用地超过 25000 sq.ft. 的比例。
# INDUS：城镇非零售商用土地的比例。
# CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
# NOX：一氧化氮浓度。
# RM：住宅平均房间数。
# AGE：1940 年之前建成的自用房屋比例。
# DIS：到波士顿五个中心区域的加权距离。
# RAD：辐射性公路的接近指数。
# TAX：每 10000 美元的全值财产税率。
# PTRATIO：城镇师生比例。
# B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
# LSTAT：人口中地位低下者的比例。

# MEDV：自住房的平均房价，以千美元计。

class Reg_Tree_Node(object):
    def __init__(self, part_feat=-1, part_v=0, c1=0, c2=0, left=None, right=None):
        self.part_feat = part_feat
        self.part_v = part_v
        self.c1 = c1
        self.c2 = c2
        self.left = left
        self.right = right

    def fit(self, part_feat=-1, part_v=0, c1=0, c2=0, left=None, right=None):
        self.part_feat = part_feat
        self.part_v = part_v
        self.c1 = c1
        self.c2 = c2
        self.left = left
        self.right = right


def feature_partition(x, y, j):
    sorted_idx = np.argsort(x[:,j])
    sorted_x = x[sorted_idx]
    sorted_y = y[sorted_idx]
    num, dim = x.shape
    mse_min = sys.float_info.max

    for i in range(num-1):
        v = (sorted_x[i][j] + sorted_x[i+1][j])/2
        y_R1 = np.sum(sorted_y[0:i+1])/(i+1)
        mse_R1 = np.sum((sorted_y[0:i+1]-y_R1)**2)
        y_R2 = np.sum(sorted_y[i+1:])/(num-i-1)
        mse_R2 = np.sum((sorted_y[i+1:]-y_R2)**2)
        mse = mse_R1 + mse_R2
        if mse < mse_min:
            mse_min = mse
            part_v = v
            c1 = y_R1
            c2 = y_R2
            R1_idx = sorted_idx[0:i+1]
            R2_idx = sorted_idx[i+1:]
    return part_v, c1, c2,  mse_min, R1_idx, R2_idx

def partition(x, y):
    mse_min = sys.float_info.max
    for j in range(x.shape[1]):
        part_v_, c1_, c2_, mse_, R1_idx_, R2_idx_ = feature_partition(x, y, j)
        if mse_ < mse_min:
            mse_min = mse_
            part_v, c1, c2, R1_idx, R2_idx = part_v_, c1_, c2_, R1_idx_, R2_idx_
            part_feat = j
    return part_feat, part_v, c1, c2, R1_idx, R2_idx, mse_min

def deep_partition(x, y, root, MSE_MIN_):
    # if part_break:
    #     return
    part_feat, part_v, c1, c2, R1_idx, R2_idx, mse = partition(x, y)

    #part_list.append((part_feat, part_v, c1, c2))

    R1_x = x[R1_idx]
    R1_y = y[R1_idx]
    R2_x = x[R2_idx]
    R2_y = y[R2_idx]

    # 若mse达到阈值, 不再继续分裂结点
    if mse < MSE_MIN_:
        root.fit(part_feat, part_v, c1, c2)
        return

    lNode = Reg_Tree_Node() if len(R1_x) >= 2 else None
    rNode = Reg_Tree_Node() if len(R2_x) >= 2 else None
    root.fit(part_feat, part_v, c1, c2, lNode, rNode)

    if lNode:
        deep_partition(R1_x, R1_y, lNode, MSE_MIN_)
    if rNode:
        deep_partition(R2_x, R2_y, rNode, MSE_MIN_)

def cart_regression_y(x, root):
    feat = root.part_feat
    part_v = root.part_v
    if x[feat] <= part_v:
        if root.left:
            return cart_regression_y(x, root.left)
        else:
            return root.c1
    else:
        if root.right:
            return cart_regression_y(x, root.right)
        else:
            return root.c2

def cart_height(root):
    if root:
        left_h = cart_height(root.left)
        right_h = cart_height(root.right)
        return 1+np.max((left_h, right_h))
    else:
        return 0

def leaves_number(root):
    if not root:
        return 0

    if root.right or root.left:
        leaves_left = leaves_number(root.left)
        leaves_right = leaves_number(root.right)
        return leaves_left+leaves_right
    else:
        return 1

def plot(y):
    plt.xlabel("idx")
    plt.ylabel("y")
    # plt.scatter(range(len(y)), y, marker='o')
    # plt.scatter(range(len(y)), y_cart, marker='x')
    for y_ in y:
        plt.plot(range(len(y_)), y_)
    plt.show()

if __name__ == '__main__':
    # 载入数据集
    Boston = datasets.load_boston()
    # print(Boston.feature_names)
    x = Boston.data  # shape:(506, 13)
    ss = StandardScaler()
    x = ss.fit_transform(x)  # 特征缩放 (x-u)/s
    y = Boston.target  # shape:(506,)

    # dim = 20
    # nums = 600
    # x = np.random.randn(dim*nums)*2
    # x = np.reshape(x, (nums, dim))
    # #x = np.repeat(x, 10, axis=1)
    # #siny = np.sin(np.sum(x, axis=1))+np.random.randn(len(x))
    # siny = np.random.randn(len(x))*10 + 30
    # #x = np.reshape(x, (len(x), 1))
    # y = siny
    # print(x)
    # print(y)

    MSE_MIN = (5, 50)

    # mse_list = []
    # for mse_min in range(MSE_MIN[0], MSE_MIN[1], 2):
    #     tree = Reg_Tree_Node()
    #     deep_partition(x, y, tree, mse_min)
    #     height = cart_height(tree)
    #     print("height: %d"%height)
    #     leaves_num = leaves_number(tree)
    #     print("leaves number: %d"%leaves_num)
    #
    #     y_cart = np.array([cart_regression_y(xi, tree) for xi in x])
    #     mse = np.sum((y-y_cart)**2)/len(y)
    #     print("mse_min: %f"%mse_min)
    #     print("total mse: %f"%mse)
    #     mse_list.append(mse)
    #     #plot((y, y_cart))
    #     tree = Reg_Tree_Node()
    #     deep_partition(x, y, tree, mse_min)
    #     height = cart_height(tree)
    #     print("height: %d"%height)
    #     leaves_num = leaves_number(tree)
    #     print("leaves number: %d"%leaves_num)
    #
    #     y_cart = np.array([cart_regression_y(xi, tree) for xi in x])
    #     mse = np.sum((y-y_cart)**2)/len(y)
    #     print("mse_min: %f"%mse_min)
    #     print("total mse: %f"%mse)
    #     mse_list.append(mse)
    #     #plot((y, y_cart))

    mse_shr = 0.00001
    tree = Reg_Tree_Node()
    deep_partition(x, y, tree, mse_shr)
    height = cart_height(tree)
    print("height: %d"%height)
    leaves_num = leaves_number(tree)
    print("leaves number: %d"%leaves_num)

    y_cart = np.array([cart_regression_y(xi, tree) for xi in x])
    mse_cart = np.sum((y-y_cart)**2)
    print("mse_min: %f"%mse_shr)
    print("total mse: %f"%mse_cart)
    #plot((y, y_cart))

    clf = DecisionTreeRegressor()
    clf.fit(x, y)
    yp = clf.predict(x)
    mse_sk = np.sum((y - yp) ** 2)
    print("total mse: %f" % mse_sk)

    plot((y, y_cart))