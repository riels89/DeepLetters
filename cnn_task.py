from docopt import docopt

from solver import CNN  # Your model.py file.

if __name__ == '__main__':
    #arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    absolute_root = '.'
    learning_rate = [.0001]
    cnn = CNN(absolute_root=absolute_root, learning_rate=learning_rate, resume=False, momentum=.9, batch_size=128, start_epoch=0)
    # Run the training job
    cnn.train()
    # cnn = CNN(absolute_root=absolute_root, learning_rate=learning_rate, resume=False, momentum=.9, batch_size=32)
    #

    # cnn.train()
