from nn.layers import FullyConnectLayer
from nn.layers import EucLoss
import load_mnist
import random
import numpy as np

class Model(object):
    def __init__(self, x_dim=784, y_dim=10):
        self.layer1 = FullyConnectLayer(x_dim, 100)
        self.layer2 = FullyConnectLayer(100, y_dim)

    def inference(self, x_input):
        a0 = x_input
        a1 = self.layer1.inference(a0)
        a2 = self.layer2.inference(a1)
        y = a2
        return y

    def accuracy(self, y_true, y_out):
        y_true_label = np.argmax(y_true, axis=1)
        y_out_label = np.argmax(y_out, axis=1)
        accuracy = np.array(np.equal(y_true_label, y_out_label), dtype=int)
        accuracy = sum(accuracy)/len(accuracy)
        return accuracy

    def train(self, x_input, y_true, epoch=50, batch_size=128):
        batch_num = (len(y_true) + batch_size - 1) // batch_size
        index = list(range(len(y_true)))
        for i in range(epoch):
            random.shuffle(index)
            for j in range(batch_num):
                start_pos = j*batch_size
                end_pos = (j+1)*batch_size
                if end_pos >= len(y_true):
                    end_pos = len(y_true)
                x_in = x_input[index[start_pos:end_pos]]
                y_t = y_true[index[start_pos:end_pos]]
                y_out = self.inference(x_in)
                accu = self.accuracy(y_t, y_out)
                loss = EucLoss.loss(y_t, y_out)
                grad_ori = EucLoss.grad_originate(y_t, y_out)
                grad_2 = self.layer2.update(grad_ori)
                self.layer1.update(grad_2)
                print("epoch:%d, batch:%d, loss:%f, accu%f"%(i, j, loss, accu))

    def test(self, test_x, test_y):
        y_out = self.inference(test_x)
        accu = self.accuracy(test_y, y_out)
        print("test accu:%f"%accu)

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist.load_mnist('D:\project\ByHand\dataset\mnist')
    model = Model()
    model.train(train_images, train_labels, epoch=5, batch_size=1000)
    model.test(test_images, test_labels)
