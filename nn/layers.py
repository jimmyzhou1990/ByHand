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

class SoftmaxLayer(object):
    def __init__(self, input_n, output_n, name='softmax', lr=0.01, alpha=0.01):
        self.input_n = input_n
        self.output_n = output_n
        self.name = name

        # 权重w 偏置b
        self.w = np.random.randn(self.output_n, self.input_n)/np.sqrt(self.input_n)
        self.b = np.zeros(self.output_n)
        self.lr = lr
        self.alpha = alpha

    def inference(self, input_a):
        z = np.dot(input_a, self.w.transpose()) + self.b
        exp_z = np.exp(z)
        exp_sum = np.expand_dims(np.sum(exp_z, axis=1), axis=1)
        exp_sum_pad = np.repeat(exp_sum, self.output_n, axis=1)
        self.output_a = exp_z/exp_sum_pad
        self.input_a = input_a
        self.batch_size = len(input_a)
        return self.output_a

    def loss_sum(self, y_true):
        pt = self.output_a[range(len(y_true)), np.argmax(y_true, axis=1)]
        pt = np.clip(pt, 1e-6, 1)
        loss = -np.log(pt)
        loss_sum = np.mean(loss)
        return loss_sum

    def grad(self, y_true):
        grad_ori = self.output_a - y_true
        return grad_ori

    def update(self, grad_z1):
        if grad_z1.shape[1] != self.output_n or grad_z1.shape[0] != self.batch_size:
            raise RuntimeError('invalid input size:[%d, %d]'%(grad_z1.shape[0], grad_z1.shape[1]))

        tmp = grad_z1
        # 回传的梯度
        grad_a0 = np.dot(tmp, self.w)

        # b的梯度
        grad_b = np.sum(tmp, axis=0)/self.batch_size

        # w的梯度
        grad_w_list = []
        for i in range(self.batch_size):
            grad_w_list.append(np.outer(tmp[i], self.input_a[i])) # 外积
        grad_w = np.array(grad_w_list)
        grad_w = np.sum(grad_w, axis=0)/self.batch_size

        #更新
        self.w = self.w*(1-self.alpha*self.lr) - self.lr*grad_w
        self.b = self.b - self.lr*grad_b
        if False:
            #print(grad_w[0])
            print(self.w[5])
        return grad_a0

class FullyConnectLayer(object):
    def __init__(self, input_n, output_n, name='layer', activation_fun='sigmoid', lr=0.01, alpha=0.01):
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
        self.lr = lr
        self.alpha = alpha

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
            grad_w_list.append(np.outer(tmp[i], self.input_a[i])) # 外积
        grad_w = np.array(grad_w_list)
        grad_w = np.sum(grad_w, axis=0)/self.batch_size
        if self.name == 'layerx':
            print(grad_w[5])
        #更新
        self.w = self.w*(1-self.alpha*self.lr) - self.lr*grad_w
        self.b = self.b - self.lr*grad_b
        return grad_a0

class MSELoss(object):
    def __init__(self):
        pass

    @staticmethod
    def loss(y_true, y_out):
        loss = np.sum(np.square(y_true - y_out))/len(y_true)
        return loss

    @staticmethod
    def grad_originate(y_true, y_out):
        return y_out - y_true

class CrossEntropy(object):
    def __init__(self):
        pass

    @staticmethod
    def loss(y_true, y_out):
        py = y_out[range(len(y_true)), np.argmax(y_true, axis=1)]
        py = np.clip(py,1e-6, 1-1e-6)
        loss = -np.log(py)
        loss = np.mean(loss)
        return loss

    @staticmethod
    def grad_originate(y_true, y_out):
        grad = -1.0/np.clip(y_out,1e-6, 1)
        grad = grad*y_true
        return grad




