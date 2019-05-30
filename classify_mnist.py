from nn.layers import FullyConnectLayer
from nn.layers import EucLoss
import load_mnist
import random
import numpy as np

class Model(object):
    def __init__(self, frame=[784, 100, 10]):
        self.layers = []
        for i in range(len(frame)-1):
            self.layers.append(FullyConnectLayer(frame[i], frame[i+1], 'layer%d'%i, activation_fun='relu'))

    def inference(self, x_input):
        a = x_input
        for i in range(len(self.layers)):
            a = self.layers[i].inference(a)
        y = a
        return y

    def accuracy(self, y_true, y_out):
        y_true_label = np.argmax(y_true, axis=1)
        y_out_label = np.argmax(y_out, axis=1)
        accuracy = np.array(np.equal(y_true_label, y_out_label), dtype=int)
        accuracy = sum(accuracy)/len(accuracy)
        return accuracy

    def train(self, x_train, y_train, x_test, y_test, epoch=50, batch_size=128):
        batch_num = (len(y_train) + batch_size - 1) // batch_size
        index = list(range(len(y_train)))
        for i in range(epoch):
            random.shuffle(index)
            for j in range(batch_num):
                start_pos = j*batch_size
                end_pos = (j+1)*batch_size
                if end_pos >= len(y_train):
                    end_pos = len(y_train)
                x_in = x_train[index[start_pos:end_pos]]
                y_t = y_train[index[start_pos:end_pos]]
                y_out = self.inference(x_in)
                accu = self.accuracy(y_t, y_out)
                loss = EucLoss.loss(y_t, y_out)
                grad_back = EucLoss.grad_originate(y_t, y_out)
                for k in range(len(self.layers)-1, -1, -1):
                    grad_back = self.layers[k].update(grad_back)
                print("epoch:%d, batch:%d, loss:%f, accu%f"%(i, j, loss, accu))

    def test(self, test_x, test_y):
        y_out = self.inference(test_x)
        accu = self.accuracy(test_y, y_out)
        print("test accu:%f"%accu)

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist.load_mnist('D:\project\ByHand\dataset\mnist')
    model = Model()
    model.train(train_images, train_labels, test_images, test_labels, epoch=5, batch_size=1000)

