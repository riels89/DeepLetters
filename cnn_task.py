from docopt import docopt

from GF_solver import CNN

if __name__ == '__main__':
    absolute_root = '.'
    learning_rate = [2]

    cnn = CNN(absolute_root=absolute_root, learning_rate=learning_rate, resume=True, momentum=.9, batch_size=1, start_epoch=0, epochs=1)
    cnn.train()
