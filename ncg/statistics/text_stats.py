import torch
import matplotlib.pyplot as plt
import numpy as np

from ncg.io.file_helpers import read_lines_multiple_files

def printWordStats(vocab):
    wordcounts = list(vocab.word2count.values())
    print(f'Number of words: {sum(wordcounts)}')
    print(f'Number of unique words: {len(wordcounts)}')
    print(f'Number of unique words after replacement with UNKNOWN: {vocab.n_words}')
    
def plotWordFrequencies(vocab, fpath_save = None):
    wordcounts = list(vocab.word2count.values())
    plt.hist(wordcounts, bins=np.logspace(np.log10(1),np.log10(100000), 50))
    plt.gca().set_xscale("log")
    plt.xlabel('word frequency')
    plt.ylabel('number of words')
    if fpath_save:
        _ = plt.savefig(fpath_save)
    plt.show()

def plotSentenceLengthFrequencies(fpaths, fpath_save = None):
    sentences = read_lines_multiple_files(fpaths)
    sentence_lengths = [len(s.split(' ')) for s in sentences]
    plt.hist(sentence_lengths)
    plt.xlabel('sentence length')
    plt.ylabel('number of sentences')
    if fpath_save:
        _ = plt.savefig(fpath_save)
    plt.show()

