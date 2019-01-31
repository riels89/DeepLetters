from docopt import docopt

from solver import CNN  # Your model.py file.

if __name__ == '__main__':
    arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    absolute_root = arguments['<root>']
    learning_rate = arguments['<learning_rates>']
    cnn = CNN(absolute_root=absolute_root, learning_rate=learning_rate)
    # Run the training job
    cnn.train()
