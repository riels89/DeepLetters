from docopt import docopt

from solver import CNN

if __name__ == '__main__':
    absolute_root = '.'
    learning_rate = [1e-6]

    cnn = CNN(absolute_root=absolute_root, learning_rate=learning_rate, resume=True, momentum=.9, batch_size=32, start_epoch=7, epochs=1)
    cnn.train()
