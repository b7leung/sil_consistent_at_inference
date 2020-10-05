import torch

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def weights_init_normal(m, var=0.0001):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
         # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,0.0001)
        # m.bias.data should be 0
        m.bias.data.fill_(0)

