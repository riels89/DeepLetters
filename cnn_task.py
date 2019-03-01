from docopt import docopt

from solver import CNN  # Your model.py file.

if __name__ == '__main__':
    #arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    absolute_root = '.'
    learning_rate = [1e-6]
    cnn = CNN(absolute_root=absolute_root, learning_rate=learning_rate, resume=True, momentum=.9, batch_size=32, start_epoch=20)
    # Run the training job
    cnn.train()
    # cnn = CNN(absolute_root=absolute_root, learning_rate=learning_rate, resume=False, momentum=.9, batch_size=32)
    #

    # cnn.train()
