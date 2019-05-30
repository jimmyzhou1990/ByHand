import numpy as np

def sigmoid(x):
    a = 1.0 / (1.0 + np.exp(-x))
    return a

def grad_sigmoid(a):
    g = a * (1.0 - a)
    return g

def relu(x):
    return np.maximum(x, 0)

def grad_relu(z):
    b = z.copy()
    b[b > 0] = 1
    return b

class FullyConnectLayer(object):
    def __init__(self, input_n, output_n, name='layer', activation_fun='sigmoid', rate=0.01):
        self.input_n = input_n
        self.output_n = output_n
        self.name = name
        if activation_fun == 'sigmoid':
            self.act_fun = sigmoid
            self.grad_act_fun = grad_sigmoid
        else:
            self.act_fun = relu
            self.grad_act_fun = grad_relu

        # 权重w 偏置b
        self.w = np.random.randn(self.output_n, self.input_n)/np.sqrt(self.input_n)
        self.b = np.zeros(self.output_n)
        self.rate = rate

    def inference(self, input_a):
        self.output_a = self.act_fun(np.dot(input_a, self.w.transpose()) + self.b)
        self.input_a = input_a
        self.batch_size = len(input_a)
        return self.output_a

    def update(self, grad_a1):
        if grad_a1.shape[1] != self.output_n or grad_a1.shape[0] != self.batch_size:
            raise RuntimeError('invalid input size:[%d, %d]'%(grad_a1.shape[0], grad_a1.shape[1]))

        tmp = grad_a1 * self.grad_act_fun(self.output_a)
        # 回传的梯度
        grad_a0 = np.dot(tmp, self.w)

        # b的梯度
        grad_b = np.sum(tmp, axis=0)/self.batch_size

        # w的梯度
        grad_w_list = []
        for i in range(self.batch_size):
            grad_w_list.append(np.outer(grad_a1[i], self.input_a[i])) # 外积
        grad_w = np.array(grad_w_list)
        grad_w = np.sum(grad_w, axis=0)/self.batch_size

        #更新
        self.w = self.w - self.rate*grad_w
        self.b = self.b - self.rate*grad_b
        return grad_a0

class EucLoss(object):
    def __init__(self):
        pass

    @staticmethod
    def loss(y_true, y_out):
        loss = np.sum(np.square(y_true - y_out))/len(y_true)
        return loss

    @staticmethod
    def grad_originate(y_true, y_out):
        return y_out - y_true






