import numpy as np

def tanh(x):
    a = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return a

def grad_tanh(a):
    return 1-a*a

def softmax(z):
    #print(z)
    shape = z.shape[0]
    exp_z = np.exp(z)
    exp_sum = np.sum(exp_z)
    exp_sum_pad = np.array([exp_sum]*shape)
    y = exp_z / exp_sum_pad
    return y

class RNN(object):
    def __init__(self, input_unit, hidden_unit, output_unit, lr=0.01, act_fun=None):
        self.input_unit = input_unit
        self.hidden_unit = hidden_unit
        self.output_unit = output_unit
        self.act_fun = act_fun
        self.W = np.random.rand(hidden_unit, hidden_unit)/hidden_unit
        self.U = np.random.rand(input_unit, hidden_unit)/input_unit
        self.V = np.random.rand(hidden_unit, output_unit)/hidden_unit
        self.b = np.zeros(self.hidden_unit)
        self.c = np.zeros(self.output_unit)
        self.batch_size = None
        self.act_fun = tanh
        self.grad_act_fun = grad_tanh
        self.h_list = None
        self.y_list = None
        self.a_list = None
        self.x_input = None
        self.lr = lr

    def inference(self, x_input):
        h_list = []
        y_list = []
        a_list = []
        h = np.zeros(self.hidden_unit)
        for i, x in enumerate(x_input):
            a = np.dot(x, self.U) + np.dot(h, self.W) + self.b
            h = self.act_fun(a)
            o = np.dot(h, self.V) + self.c
            y = softmax(o)
            h_list.append(h)
            y_list.append(y)
            a_list.append(a)
        self.h_list = np.array(h_list)
        self.y_list = np.array(y_list)
        self.a_list = np.array(a_list)
        self.x_input = x_input
        return self.y_list, self.y_list[-1]

    def loss(self, y_true):
        loss_sum = .0
        #print(y_true.shape[0], self.h_list.shape[0])
        for t in range(y_true.shape[0]):
            #print(y_true[t])
            max_idx = np.argmax(y_true[t])
            pt = np.clip(self.y_list[t][max_idx], 1e-6, 1)
            los = -np.log(pt)
            loss_sum += los
        #print(loss_sum)
        return loss_sum

    def bptt(self, y_true):
        T = self.h_list.shape[0]
        grad_ht_L_list = np.zeros(shape=(T, self.hidden_unit))
        grad_ot_Lt_list = np.zeros(shape=(T, self.output_unit))
        for t in range(T):
            grad_ot_Lt_list[t] = self.y_list[t] - y_true[t]

        grad_ht_L_list[T-1] = np.dot(grad_ot_Lt_list[T-1], np.transpose(self.V))
        for t in range(T-2, -1, -1):
            grad_ht_L_list[t] = np.dot(grad_ht_L_list[t+1]*self.grad_act_fun(self.a_list[t+1]), np.transpose(self.W))+\
                np.dot(grad_ot_Lt_list[t], np.transpose(self.V))

        grad_W_L = np.zeros((self.hidden_unit, self.hidden_unit))
        grad_U_L = np.zeros((self.input_unit, self.hidden_unit))
        grad_V_L = np.zeros((self.hidden_unit, self.output_unit))
        grad_b_L = np.zeros(self.hidden_unit)
        grad_c_L = np.zeros(self.output_unit)
        for t in range(T):
            if t > 0:
                grad_W_L += np.outer(grad_ht_L_list[t]*self.grad_act_fun(self.a_list[t]), self.h_list[t-1])
            grad_U_L += np.outer(self.x_input[t], grad_ht_L_list[t]*self.grad_act_fun(self.a_list[t]))
            grad_V_L += np.outer(self.h_list[t], grad_ot_Lt_list[t])
            grad_b_L += grad_ht_L_list[t]*self.grad_act_fun(self.a_list[t])
            grad_c_L += grad_ot_Lt_list[t]

        #print(grad_W_L)
        self.U = self.U - self.lr*grad_U_L
        self.W = self.W - self.lr*grad_W_L
        self.V = self.V - self.lr*grad_V_L
        self.b = self.b - self.lr*grad_b_L
        self.c = self.c - self.lr * grad_c_L