import numpy as np
import random
import re

size = 26

def yield_sample(path):
    f = open(path, 'r')
    all_lines = f.readlines()
    f.close()
    word_list = []
    embedding = np.eye(size+1)
    for line in all_lines:
        line = re.sub('[^a-zA-Z ]', '', line)
        words = line.strip().lower().split(' ')
        for w in words:
            if len(w) > 0:
                word_list.append(w)

    # print(len(word_list))
    while True:
        rand_idx = random.randint(0, len(word_list))
        word = word_list[rand_idx]
        #print(rand_idx, word)
        x = []
        for alph in word:
            idx = ord(alph) - ord('a')
            x.append(embedding[idx])

        y = x[1:]
        y.append(embedding[size])
        yield word, np.array(x), np.array(y)

def yield_sample_(path, seq_length):
    data = open(path, 'r').read()  # should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    embedding = np.eye(vocab_size)

    while True:
        rand_idx = random.randint(0, len(data)-seq_length-1)
        seq = data[rand_idx:rand_idx+seq_length+1]
        x = np.zeros((seq_length, vocab_size))
        y = np.zeros((seq_length, vocab_size))
        for idx, s in enumerate(seq[:-1]):
            x[idx] = embedding[char_to_idx[s]]
        for idx, s in enumerate(seq[1:]):
            y[idx] = embedding[char_to_idx[s]]
        yield x, y, seq

if __name__ == '__main__':
    g = yield_sample('D:\project\ByHand\dataset\input.txt')
    for i in range(10):
        word, x, y = next(g)
        print(word)
