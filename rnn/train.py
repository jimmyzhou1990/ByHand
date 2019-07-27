from rnn import RNN
from data_loader import yield_sample

def train_rnn(step, data_path):
    rnn = RNN(27, 50, 27, lr=0.01)
    gen = yield_sample(data_path)
    for i in range(step):
        word, x, y = next(gen)
        #print(y)
        rnn.inference(x)
        los = rnn.loss(y)
        rnn.bptt(y)
        print("step:%d, loss:%f"%(i, los))

if __name__ == '__main__':
    train_rnn(10000, 'D:\project\ByHand\dataset\input.txt')