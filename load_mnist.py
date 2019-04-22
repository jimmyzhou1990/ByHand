
import numpy as np
import struct
import os
import matplotlib.pyplot as plt

def load_mnist(path):
    train_label_path = os.path.join(path, 'train-labels.idx1-ubyte')
    train_image_path = os.path.join(path, 'train-images.idx3-ubyte')
    test_label_path = os.path.join(path, 't10k-labels.idx1-ubyte')
    test_image_path = os.path.join(path, 't10k-images.idx3-ubyte')

    label = np.eye(10)

    with open(train_label_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        train_labels = np.fromfile(f, dtype=np.uint8)
        train_labels = label[train_labels]

    with open(train_image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        print(magic, num, rows, cols)
        train_images = np.fromfile(f, dtype=np.uint8).reshape(len(train_labels), 784)

    with open(test_label_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        test_labels = np.fromfile(f, dtype=np.uint8)
        test_labels = label[test_labels]

    with open(test_image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        test_images = np.fromfile(f, dtype=np.uint8).reshape(len(test_labels), 784)

    print('load train_images:%d, test_images:%d'%(len(train_labels), len(test_labels)))
    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    path = 'D:\project\ByHand\dataset\mnist'
    train_images, train_labels, test_images, test_labels = load_mnist(path)
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = train_images[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()