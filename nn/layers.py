import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

class FullyConnectLayer(object):
    def __init__(self, input_n, output_n, name='layer', activation_fun='sigmoid', rate=0.1):
        self.input_n = input_n
        self.output_n = output_n
        self.name = name
        if activation_fun == 'sigmoid':
            self.act_fun = sigmoid

        # 权重w 偏置b
        self.w = np.random.randn(self.input_n, self.output_n)/np.sqrt(self.input_n)
        self.b = np.zeros(self.output_n)
        self.rate = rate

    def inference(self, input_a):
        self.output_a = self.act_fun(np.dot(input_a, self.w) + self.b)
        self.input_a = input_a
        self.batch_size = len(input_a)
        return self.output_a

    def update(self, grad_a1):
        if grad_a1.shape[1] != self.output_n or grad_a1.shape[0] != self.batch_size:
            raise RuntimeError('invalid input size:[%d, %d]'%(grad_a1.shape[0], grad_a1.shape[1]))

        # 回传的梯度
        grad_a0 = np.dot(grad_a1*self.output_a*(1-self.output_a), self.w.transpose())

        # w,b的梯度
        tmp = grad_a1*self.output_a*(1-self.output_a)
        grad_b = np.sum(tmp, axis=0)/self.batch_size
        tmp = tmp.repeat(self.input_n, axis=0)
        tmp1 = self.input_a.reshape(-1, 1).repeat(self.output_n, axis=1)
        grad_w = (tmp*tmp1).reshape(self.batch_size, self.input_n, self.output_n)
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






