import numpy as np
import copy
import time
import signal

import matplotlib.pyplot as plt
from graphviz import Digraph

from pandas import DataFrame
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, log_loss

import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.utils.data

from pycuda import autoinit, driver


#######################################################################################################################

def gpu_stat():
    if torch.cuda.is_available():

        def pretty_bytes(bytes, precision=1):
            abbrevs = ((1<<50, 'PB'),(1<<40, 'TB'),(1<<30, 'GB'),(1<<20, 'MB'),(1<<10, 'kB'),(1, 'bytes'))
            if bytes == 1:
                return '1 byte'
            for factor, suffix in abbrevs:
                if bytes >= factor:
                    break
            return '%.*f%s' % (precision, bytes / factor, suffix)

        device = autoinit.device
        print()
        print( 'GPU Name: %s' % device.name())
        print( 'GPU Memory: %s' % pretty_bytes(device.total_memory()))
        print( 'CUDA Version: %s' % str(driver.get_version()))
        print( 'GPU Free/Total Memory: %d%%' % ((driver.mem_get_info()[0] /driver.mem_get_info()[1]) * 100))

#####################################################################################################################

class HYPERPARAMETERS(dict):
    """
    Class that holds a set of hyperparameters as name-value pairs and for convenience makes them accesssable
    as attributes.
    Example:
        H = HYPERPARAMETERS({ 'parameter_name' : parameter_value, ... })
        access using H.parameter_name or by H['parameter_name']
    """
    def __init__(self, dictionary):
        super(HYPERPARAMETERS, self).__init__(dictionary)
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __getstate__(self):
        return self
    def __setstate__(self, d):
        self = d

#####################################################################################################################

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x, async=False):
    if torch.cuda.is_available():
        x = x.cuda(async=async)
    return Variable(x)

#####################################################################################################################

#save_checkpoint({
#            'model' : type(model).__name__
#            'epoch': epoch + 1,
#            'state_dict': model.state_dict(),
#            'optimizer' : optimizer.state_dict(),
#        })
def save_checkpoint(state, filename='./chkp/checkpoint.tar'):
    torch.save(state, filename)
    print("=> saved checkpoint '{}' (epoch {})".format(filename, state['epoch']))


def load_checkpoint(model, optimizer=None, filename='./chkp/checkpoint.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    return checkpoint

#####################################################################################################################

# https://github.com/robintibor/braindecode/blob/master/braindecode/torch_ext/init.py

def glorot_weight_zero_bias(model):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, 'weight') and not module.weight is None:
            if not ('BatchNorm' in module.__class__.__name__):
                init.xavier_uniform_(module.weight, gain=1)
            else:
                init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                init.constant_(module.bias, 0)

#####################################################################################################################

# https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/4

def adjust_learning_rate(optimizer, epoch, init_lr=0.001, lr_decay_epoch=30):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay_epoch epochs"""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

    return lr

#####################################################################################################################

class PlotLosses(object):
    """
    Plots the train loss and validation score in a matlibplot plot during training whenever
    plot_loss and plot_prediction are called inside the epoch loop.
    """
    def __init__(self, figsize=(12,6)):
        plt.ion()

        self.fig, self.ax = plt.subplots(1,2,figsize=figsize)

        self.ax[0].set_xlabel("Epoch");
        self.ax[0].set_ylabel("Loss");
        self.ax[0].grid(True)

        self.ax[1].set_xlabel("X");
        self.ax[1].set_ylabel("Y");
        self.ax[1].set_ylim([-1,1])
        self.ax[1].grid(True)

        self.fig.canvas.draw()
        plt.show(block=False)

    def plot_loss(self, loss, epoch, epochs, lr):
        iter_array = np.arange(1, epochs, 1)

        self.ax[0].set_title("Epoch # " + str(epoch) + "/" + str(epochs)
                             + " Loss # %.4f" % loss[-1] + " LR # %.2e" % lr)
        self.ax[0].plot(np.arange(0, epoch, 1), loss)

        self.fig.canvas.draw()
        plt.show(block=False)

    def plot_prediction(self, x, y, x_h, y_h, label1="train", label2="predict"):
        plt.cla()
        self.ax[1].set_title("Prediction")
        self.ax[1].scatter (x, y, c='red', label=str(label1), s=1.0, alpha=0.3)
        self.ax[1].scatter (x_h, y_h, c='green', label=str(label2), s=10.0, alpha=0.3)
        self.ax[1].legend()

        self.fig.canvas.draw()
        plt.show(block=False)

    def close(self):
        plt.ioff ()
        plt.close(self.fig)

#####################################################################################################################

# https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="8,8"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

#####################################################################################################################

def layer_weight(data):
    mean = np.mean(data)
    std = np.std(data)

    hist, bins = np.histogram(data, bins=50)
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    return { 'mean':mean,
             'std':std,
             'hist':hist,
             'center':center,
             'width':width
             }

def layer_stats(model):
    for module in model.modules():
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            fig, ax = plt.subplots(1,2,figsize=(8,3))

            if not module.weight is None:
                w = layer_weight(to_np(module.weight))
                ax[0].set_title("Weight - Mean # %.4f" % w['mean'] + " STD # %.2e" % w['std'])
                ax[0].bar(w['center'], w['hist'], align='center', width=w['width'])

            if not module.bias is None:
                b = layer_weight(to_np(module.bias))
                ax[1].set_title("Bias - Mean # %.4f" % b['mean'] + " STD # %.2e" % b['std'])
                ax[1].bar(b['center'], b['hist'], align='center', width=b['width'])
                plt.show()

#####################################################################################################################

# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
# https://github.com/fchollet/keras/blob/master/keras/utils/layer_utils.py

def print_summary(model, line_length=None, positions=None, print_fn=print):
    """Prints a summary of a model.
    # Arguments
        model: model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
    """

    line_length = line_length or 65
    positions = positions or [.45, .85, 1.]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Shape', 'Param #']

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print_fn(line)

    print_fn( "Summary for model: " + model.__class__.__name__)
    print_fn('_' * line_length)
    print_row(to_display, positions)
    print_fn('=' * line_length)

    def print_module_summary(name, module):
        count_params = sum([np.prod(p.size()) for p in module.parameters()])
        output_shape = tuple([tuple(p.size()) for p in module.parameters()])
        cls_name = module.__class__.__name__
        fields = [name + ' (' + cls_name + ')', output_shape, count_params]
        print_row(fields, positions)

    module_count = len(set(model.modules()))
    for i, item in enumerate(model.named_modules()):
        name, module = item
        cls_name = str(module.__class__)
        if not 'torch' in cls_name or 'container' in cls_name:
            continue

        print_module_summary(name, module)
        if i == module_count - 1:
            print_fn('=' * line_length)
        else:
            print_fn('_' * line_length)

    trainable_count = 0
    non_trainable_count = 0
    for name, param in model.named_parameters():
        if 'bias' in name or 'weight' in name :
            trainable_count += np.prod(param.size())
        else:
            non_trainable_count += np.prod(param.size())

    print_fn('Total params:         {:,}'.format(trainable_count + non_trainable_count))
    print_fn('Trainable params:     {:,}'.format(trainable_count))
    print_fn('Non-trainable params: {:,}'.format(non_trainable_count))
    print_fn('_' * line_length)

#####################################################################################################################

def stat_summary(y, y_hat, n, p):
    # local time & date
    t = time.localtime()

    # printing output to screen
    print( '\n==============================================================================' )
    print(  "Date: ", time.strftime("%a, %d %b %Y",t) )
    print(  "Time: ", time.strftime("%H:%M:%S",t) )
    print( '==============================================================================')
    print( 'Parameters:               % 5.0f' % p + '           Cases: %5.0f' % n )
    print( '==============================================================================' )
    print( 'Models stats' )
    print( '==============================================================================' )
    print( 'Mean Squared Error        % -5.6f         ' % mean_squared_error(y, y_hat) )
    print( 'Mean Absolute Error       % -5.6f         ' % mean_squared_error(y, y_hat) )
    print( 'Root Mean Squared Error   % -5.6f         ' % np.sqrt(mean_squared_error(y, y_hat)) )
    print( 'R-squared                 % -5.6f         ' % r2_score(y, y_hat) )
    print( '==============================================================================')

#####################################################################################################################

# https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py

class DelayedKeyboardInterrupt(object):
    def __init__(self):
        self.signal_received = None

    def __enter__(self):
        self.signal_received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

#####################################################################################################################

def residual_plots(y, y_hat, figsize=(16,4)):
    residuals = DataFrame(y - y_hat)

    fig, ax = plt.subplots(1,3,figsize=figsize)

    # scatter plot
    ax[0].scatter(y_hat, residuals,s=1.0, alpha=0.3)
    ax[0].set_xlabel("Predicted Y")
    ax[0].set_ylabel("Residual")
    ax[0].set_title("Scatter Plot")

    # histogram plot
    residuals.plot(kind='hist', ax=ax[1])
    ax[1].set_title('Histogram Plot')
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel('Residuals')

    # density plot
    residuals.plot(kind='kde', ax=ax[2])
    ax[2].set_title('Density Plot')

#####################################################################################################################

def plot_prediction(x, y, x_h, y_h, title="", label1="train", label2="validate"):
    plt.cla()
    plt.title(title)
    a, b = zip(*sorted(zip(x,y)))
    c, d = zip(*sorted(zip(x_h,y_h)))

    plt.plot (a, b, c='red', label=str(label1))
    plt.plot (c, d, c='green', label=str(label2))
    plt.legend()



#######################################################################################################################

class Stopping(object):
    """
    Class implement some of regularization techniques to avoid over-training as described in
    http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    """
    def __init__(self, model, patience=50):
        self.model = model
        self.patience = patience

        self.best_score = -1
        self.best_score_epoch = 0
        self.best_score_model = None
        self.best_score_state = None

    def step(self, epoch, train_score, valid_score):
        if valid_score > self.best_score:
            self.best_score = valid_score
            self.best_score_epoch = epoch
            self.best_score_state = self.model.state_dict()
            return False
        elif self.best_score_epoch + self.patience < epoch:
            return True

    def state_dict(self):
        return {
            'patience' : self.patience,
            'best_score' : self.best_score,
            'best_score_epoch' : self.best_score_epoch,
            'best_score_model' : self.best_score_model,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.best_score  = state_dict['best_score']
        self.best_score_epoch = state_dict['best_score_epoch']
        self.best_score_model = state_dict['best_score_model']

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Patience: {}\n'.format(self.patience)
        fmt_str += '    Best Score: {:.4f}\n'.format(self.best_score)
        fmt_str += '    Epoch of Best Score: {}\n'.format(self.best_score_epoch)
        return fmt_str

#######################################################################################################################