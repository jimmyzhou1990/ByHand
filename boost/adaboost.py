import numpy as np
from svm.run_svm2d import load_data_2d_quadratic
import matplotlib.pyplot as plt
import sys

from dt.cart_classification import *

class WeakClassification(object):
    def __init__(self):
        self.tree = TreeNode()  #一个根结点 两个叶子结点的cart分类树
        self.alpha = 0
        self.w = None

    def part_feature(self, x, y, w, j):
        candi_set = set(x[:, j])
        e_min = sys.float_info.max
        #print(candi_set)
        for cand in candi_set:
            lidx = np.argwhere(x[:,j]<=cand)[:,0]
            ridx = np.argwhere(x[:,j]>cand)[:,0]

            if len(lidx) == 0:
                main_lable_l = main_label(y)
                err_l = 0
            else:
                main_lable_l = main_label(y[lidx])
                y_l = np.ones(len(lidx))*main_lable_l
                err_l = np.sum((y_l!=y[lidx]).astype(np.int32)*w[lidx])

            if len(ridx) == 0:
                main_lable_r = main_label(y)
                err_r = 0
            else:
                main_lable_r = main_label(y[ridx])
                y_r = np.ones(len(ridx))*main_lable_r
                err_r = np.sum((y_r != y[ridx]).astype(np.int32)*w[ridx])

            #print(j, cand,"====>", err_l+err_r)
            if err_l+err_r<e_min:
                e_min = err_l+err_r
                part_v = cand
                D1_idx = lidx
                y_D1 = main_lable_l
                D2_idx = ridx
                y_D2 = main_lable_r
        #print(e_min)
        return part_v, D1_idx, y_D1, D2_idx, y_D2, e_min

    def partition(self, x, y, w):
        num, dim = x.shape
        e_min = sys.float_info.max
        for j in range(dim):
            part_v_, D1_idx_, y_d1_, D2_idx_, y_d2_, e = self.part_feature(x, y, w, j)
            if e < e_min:
                e_min = e
                part_feat = j
                part_v = part_v_
                D1_idx = D1_idx_
                y_D1 = y_d1_
                D2_idx = D2_idx_
                y_D2 = y_d2_
        return part_feat, part_v, D1_idx, y_D1, D2_idx, y_D2, e_min

    def train(self, x, y, w):
        part_feat, part_v, D1_idx, y_D1, D2_idx, y_D2, e_min = self.partition(x, y, w)
        print(part_feat, part_v, e_min)
        lNode = TreeNode()
        lNode.isLeaf = True
        lNode.y = y_D1
        #print(y_D1)
        rNode = TreeNode()
        rNode.isLeaf = True
        rNode.y = y_D2
        #print(y_D2)
        self.tree.fit(part_feat, part_v, lNode, rNode)

        self.alpha = 0.5*np.log((1-e_min)/e_min)
        #print(e_min, self.alpha, part_feat, part_v)
        self.w = w

        y_pred = np.zeros(len(y))
        y_pred[D1_idx] = y_D1
        y_pred[D2_idx] = y_D2

        tmp = w * np.exp(-y_pred * y * self.alpha)
        z = np.sum(tmp)
        w_forward = tmp/z
        return w_forward

    def predict(self, x):
        if x[self.tree.part_feat] <= self.tree.part_v:
            return self.alpha*self.tree.lNode.y
        else:
            return self.alpha*self.tree.rNode.y

class AdaBoostTree(object):
    def __init__(self, num=10):
        self.num = num
        self.Trees = []

    def train(self, x, y):
        w = np.ones(len(x))/len(x)
        for i in range(self.num):
            #print(w)
            weak = WeakClassification()
            w = weak.train(x, y, w)
            self.Trees.append(weak)

    def predict(self,x):
        y = 0
        for i in range(self.num):
            y += self.Trees[i].predict(x)

        return -1 if y<=0 else 1

# 圆
def load_data_2d_circle(O, R, n, data_range=[-10, 10], gap=1):
    wid = data_range[1] - data_range[0]
    x_pos = []
    np.random.seed(0)
    while len(x_pos) < n:
        x, y = np.random.rand(2)*wid + data_range[0]
        if np.power(x-O[0], 2)+np.power(y-O[1], 2) < R**2:
            x_pos.append([x, y])
    y_pos = np.zeros(len(x_pos)) + 1

    x_neg = []
    while len(x_neg) < n:
        x, y = np.random.rand(2)*wid + data_range[0]
        if np.power(x-O[0], 2)+np.power(y-O[1], 2) > R**2+gap:
            x_neg.append([x, y])
    y_neg = np.zeros(len(x_pos))-1

    x_out = np.array(x_pos + x_neg)
    y_out = np.concatenate((y_pos, y_neg))

    return x_out, y_out

if __name__ == "__main__":
    x, y = load_data_2d_circle((0,0), 5, 200, [-10, 10], 0.5)
    #plot_2d_quadratic(x, y)

    adaBoostTree = AdaBoostTree(num=83)
    adaBoostTree.train(x, y)

    y_p = np.array([adaBoostTree.predict(xi) for xi in x])
    accu = np.sum((y_p == y).astype(np.int32)) / len(y)

    print(accu)

