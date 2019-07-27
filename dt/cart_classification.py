import numpy as np
from svm.run_svm2d import load_data_2d_quadratic
import matplotlib.pyplot as plt
import sys
from boost.adaboost import *

class TreeNode(object):
    def __init__(self, part_feat=-1, part_v=0, lNode=None, rNode=None, y=0, isLeaf=False):
        self.part_feat = part_feat
        self.part_v = part_v
        self.lNode = lNode
        self.rNode = rNode
        self.y = y
        self.isLeaf = isLeaf

    def fit(self, part_feat=-1, part_v=0, lNode=None, rNode=None, y=0, isLeaf=False):
        self.part_feat = part_feat
        self.part_v = part_v
        self.lNode = lNode
        self.rNode = rNode
        self.y = y
        self.isLeaf = isLeaf

def part_feature(x, y, j):
    #print(x[:,j])
    sorted_idx = np.argsort(x[:,j])
    sorted_y = y[sorted_idx]
    sorted_x = x[sorted_idx]
    num, dim = x.shape
    idx = 0

    gini_min = sys.float_info.max
    while idx<num:
        i = idx
        while i<num and sorted_x[idx][j] == sorted_x[i][j]:
            i += 1

        y_D1 = sorted_y[0:i]
        D1_p = np.sum((y_D1>0).astype(np.float))/len(y_D1)
        gini_D1 = 2 * D1_p * (1-D1_p)

        if i >= num:
            gini_D2 = 0
        else:
            y_D2 = sorted_y[i:]
            D2_p = np.sum((y_D2>0).astype(np.float))/len(y_D2)
            gini_D2 = 2 * D2_p * (1 - D2_p)

        gini = (i/num)*gini_D1 + ((num-i)/num)*gini_D2
        #print(gini)
        if gini < gini_min:
            gini_min = gini
            if i >= num:
                part_v = sorted_x[i-1][j]
            else:
                part_v = (sorted_x[i-1][j]+sorted_x[i][j])/2
            D1_idx = sorted_idx[0:i]
            D2_idx = sorted_idx[i:]
        idx = i
    return part_v, D1_idx, D2_idx, gini_min

def partition(x, y):
    num, dim = x.shape
    gini_min = sys.float_info.max
    for j in range(dim):
        part_v_, D1_idx_, D2_idx_, gini = part_feature(x, y, j)
        #print(gini)
        if gini < gini_min:
            gini_min = gini
            part_feat = j
            part_v = part_v_
            D1_idx = D1_idx_
            D2_idx = D2_idx_
    return part_feat, part_v, D1_idx, D2_idx, gini_min

def main_label(y):
    pos_num = np.sum((y>0).astype(np.float))
    neg_num = len(y) - pos_num
    label = 1 if pos_num>neg_num else -1
    return label

def create_tree(root, x, y):
    mlabel = main_label(y)

    # 基尼系数小于阈值,停止划分,设为叶结点
    p = np.sum((y > 0).astype(np.float)) / len(y)
    gini = 2 * p * (1 - p)
    if gini < GINI_MIN:
        print("gini: %f, no need to partition"%gini)
        # print(y)
        # print(mlabel)
        root.y = mlabel
        root.isLeaf = True
        return

    # 样本点个数小于阈值, 停止划分,设为叶结点
    if len(y) < LEAVES_MIN:
        root.y = mlabel
        root.isLeaf = True
        return

    part_feat, part_v, D1_idx, D2_idx, gini_ = partition(x, y)
    print(part_feat, part_v, len(D1_idx), len(D2_idx), gini_)
    lNode = TreeNode()
    rNode = TreeNode()
    root.fit(part_feat, part_v, lNode, rNode)

    if len(D1_idx) == 0:
        lNode.isLeaf = True
        lNode.y = mlabel
    else:
        create_tree(lNode, x[D1_idx], y[D1_idx])

    if len(D2_idx) == 0:
        rNode.isLeaf = True
        rNode.y = mlabel
    else:
        create_tree(rNode, x[D2_idx], y[D2_idx])

def predict(root, x):
    if root.isLeaf:
        return root.y
    if x[root.part_feat] <= root.part_v:
        return predict(root.lNode, x)
    else:
        return predict(root.rNode, x)

def leaves_number(root):
    if not root:
        return 0

    if root.rNode or root.lNode:
        leaves_left = leaves_number(root.lNode)
        leaves_right = leaves_number(root.rNode)
        return leaves_left+leaves_right
    else:
        return 1

def plot_2d_quadratic(data, label):
    plt.xlabel("x")
    plt.ylabel("y")
    #print(label)
    for idx, (xi, label) in enumerate(zip(data, label)):
        if label > 0:
            plt.plot(xi[0], xi[1], 'ro')
        else:
            plt.plot(xi[0], xi[1], 'gx')
    plt.yticks([i for i in range(-10, 10)])
    plt.xticks([i for i in range(-10, 10)])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    #x, y = load_data_2d_quadratic(0.1, -3, 100, [-10, 10], 0.5)
    x, y = load_data_2d_circle((0, 0), 5, 100, [-10, 10], 0.5)
    #plot_2d_quadratic(x, y)

    GINI_MIN = 0.00001
    LEAVES_MIN = 2

    tree = TreeNode()
    create_tree(tree, x, y)

    y_p = np.array([predict(tree, xi) for xi in x])

    accu = np.sum((y_p == y).astype(np.int32))/len(y)

    print(accu)
    print(leaves_number(tree))
    #plot_2d_quadratic(x, y_p)