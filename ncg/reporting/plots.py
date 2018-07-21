import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plotBleuScores(epoch_intervals, bleu_scores, fname = None):    
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.ticklabel_format(axis='x', scilimits=(0, 0))
    plt.plot(epoch_intervals, bleu_scores, 'ro-', color='blue', label='BLEU on validation set after epoch')
    plt.xlabel('#training pairs')
    plt.ylabel('BLEU')
    plt.legend()
    if fname:
        _ = plt.savefig(fname)
    else:
        plt.show()

def plotEpochLosses(intervals_train, losses_train, intervals_val, losses_val, fname = None):    
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.ticklabel_format(axis='x', scilimits=(0, 0))
    if losses_train:
        plt.plot(intervals_train, losses_train, 'ro-', color='blue', label='train loss over epoch')
    if losses_val:
        plt.plot(intervals_val, losses_val, 'ro-', color='red', label='validation loss after epoch')
    plt.xlabel('#training pairs')
    plt.ylabel('average token loss')
    plt.legend()
    if fname:
        _ = plt.savefig(fname)
    else:
        plt.show()

# TODO: extract building the frame
def plotBatchLosses(intervals, losses, fname = None):
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.ticklabel_format(axis='x', scilimits=(0, 0))
    plt.plot(intervals, losses, color='blue', label='train loss over batch')
    plt.xlabel('#training pairs')
    plt.ylabel('average token loss')
    plt.legend()
    if fname:
        _ = plt.savefig(fname)
    else:
        plt.show()

