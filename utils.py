import os
import codecs
import pdb

import torch
import numpy as np
from functools import reduce
import operator
import cfg
import re
import numpy as np
from collections import Counter
from math import ceil
from math import floor
from math import log
import random
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist


def describe(t):  # t could be numpy or torch tensor.
    t = t.data if isinstance(t, torch.autograd.Variable) else t
    s = '{:17s} {:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
    ttype = 'np.{}'.format(t.dtype) if type(t) == np.ndarray else str(t.type()).replace('ensor', '')
    si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
    return s.format(ttype, si, t.min(), t.max(), t.mean(), t.std())


def write_gen_samples(samples, fn, c_lab=None):
    """ samples: list of strings. c_lab (optional): tensor of same size. """
    fn_dir = os.path.dirname(fn)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)

    size = len(samples)
    with open(fn, 'w+') as f:
        if c_lab is not None:
            print("Saving %d samples with labels" % size)
            assert c_lab.nelement() == size, 'sizes dont match'
            f.writelines(['label: {}\n{}\n'.format(y, s) for y, s in zip(c_lab, samples)])
        else:
            print("Saving %d samples without labels" % size)
            f.write('\n'.join(samples) + '\n')


def write_interpsamples(samples, fn, c_lab=False):
    raise Exception('Reimplement this function like write_gen_samples(), use minibatch')


def write_samezsamples(samples, samples2, fn, fn2, lab=False):
    raise Exception('Reimplement this function like write_gen_samples(), use minibatch')


def save_vocab(vocab, fn):
    check_dir_exists(fn)
    with codecs.open(fn, "w", "utf-8") as f:
        for word, ix in vocab.stoi.items():
            f.write(word + " " + str(ix) + "\n")
    print('Saved vocab to ' + fn)


# Linearly interpolate between start and end val depending on current iteration
def interpolate(start_val, end_val, start_iter, end_iter, current_iter):
    if current_iter < start_iter:
        return start_val
    elif current_iter >= end_iter:
        return end_val
    else:
        return start_val + (end_val - start_val) * (current_iter - start_iter) / (end_iter - start_iter)


def anneal(cfgan, it):
    return interpolate(cfgan.start.val, cfgan.end.val, cfgan.start.iter, cfgan.end.iter, it)


def check_dir_exists(fn):
    fn_dir = os.path.dirname(fn)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def scale_and_clamp(dist, w, clamp_val=None):
    rescaled = dist * w  # w = 1/scale
    if clamp_val and rescaled > clamp_val:
        return clamp_val
    else:
        return rescaled


def idx2sentence(seq):
    peptide=''
    has_unk = False
    for i in range(seq.size(0)):
        if seq[i] == 0:
            has_unk = True
        if seq[i] not in [1,2,3]:
            peptide+=cfg.AA_abb_dict_T[int(seq[i])]
    return peptide, has_unk


# compare single base
def SingleBaseCompare(seq1, seq2, i, j):
    if seq1[i] == seq2[j]:
        return 2
    else:
        return -1


# Smith–Waterman Alignment
def SMalignment(seq1, seq2):
    m = len(seq1)
    n = len(seq2)
    if min(m, n) == 0:
        return 0
    g = -3
    matrix = []
    for i in range(0, m):
        tmp = []
        for j in range(0, n):
            tmp.append(0)
        matrix.append(tmp)
    for sii in range(0, m):
        matrix[sii][0] = sii * g
    for sjj in range(0, n):
        matrix[0][sjj] = sjj * g
    for siii in range(1, m):
        for sjjj in range(1, n):
            matrix[siii][sjjj] = max(matrix[siii - 1][sjjj] + g,
                                     matrix[siii - 1][sjjj - 1] + SingleBaseCompare(seq1, seq2, siii, sjjj),
                                     matrix[siii][sjjj - 1] + g)
    sequ1 = [seq1[m - 1]]
    sequ2 = [seq2[n - 1]]
    while m > 1 and n > 1:
        if max(matrix[m - 1][n - 2], matrix[m - 2][n - 2], matrix[m - 2][n - 1]) == matrix[m - 2][n - 2]:
            m -= 1
            n -= 1
            sequ1.append(seq1[m - 1])
            sequ2.append(seq2[n - 1])
        elif max(matrix[m - 1][n - 2], matrix[m - 2][n - 2], matrix[m - 2][n - 1]) == matrix[m - 1][n - 2]:
            n -= 1
            sequ1.append('-')
            sequ2.append(seq2[n - 1])
        else:
            m -= 1
            sequ1.append(seq1[m - 1])
            sequ2.append('-')
    sequ1.reverse()
    sequ2.reverse()
    align_seq1 = ''.join(sequ1)
    align_seq2 = ''.join(sequ2)
    align_score = 0.
    for k in range(0, len(align_seq1)):
        if align_seq1[k] == align_seq2[k]:
            align_score += 1
    align_score = float(align_score) / len(align_seq1)
    return align_score


class Vocab:
    '''
    Wrapper for ix2word and word2ix for converting sequences
    '''

    def __init__(self):
        self.fix_length = cfg.pep_max_length_pepbdb
        self.ix2word = {}
        self.word2ix = {}
        for key in cfg.AA_abb_dict:
            word = " " + key
            self.ix2word[cfg.AA_abb_dict[key]] = word
            self.word2ix[key] = cfg.AA_abb_dict[key]

        self.special_tokens = set(['<unk>', '<pad>', '<start>', '<eos>'])
        self.special_tokens_ix = {self.word2ix[w] for w in self.special_tokens}

    def to_ix(self, seq, fix_length=True):
        if type(seq) == str:
            seq = seq.split()
        elif type(seq) == list:
            seq = seq
        else:
            raise ValueError('Only strings or lists of strings accepted.')
        # Make sure to have BOS and EOS symbols
        if seq[0] != "<start>":
            seq = ["<start>"] + seq
        if seq[-1] != "<eos>":
            seq = seq + ["<eos>"]
        # optionally pad seq to fix_length
        if fix_length:
            num_pads = self.fix_length - len(seq)
            seq = seq + ["<pad>"] * num_pads

        seq_ix = [self.word2ix[tok] for tok in seq]
        seq_ix = torch.LongTensor(seq_ix).view(1, -1)
        return seq_ix

    def to_word(self, seq, print_special_tokens=True):
        seq = [s.item() for s in seq]
        if not print_special_tokens:
            seq = [i for i in seq if not i in self.special_tokens_ix]
        return [self.ix2word[s] for s in seq]

    def size(self):
        return len(self.ix2word)


def sample_from_model(model,
                      vocab,
                      z=None,
                      c=None,
                      n_samples=2,
                      print_special_tokens=True,
                      **sample_kwargs):
    '''
    Wrapper for the generate_sentence function of the model
    params:
        model: model object
        z: latent space (will be sampled if not specified)
            hid_size x num_samples
        c: condition (will also be sampled if not specified)
            1 x num_samples
        sample_mode: how to generate
    '''
    # vocab_itos = vocab[0] # itos, stoi -> only need itos

    samples, z, c = model.generate_sentences(
        n_samples, z=z, **sample_kwargs)

    if sample_kwargs['sample_mode'] == 'beam':
        predictions = [[vocab.to_word(s_topK, print_special_tokens)
                        for s_topK in s] for s in samples]
    else:
        predictions = [[vocab.to_word(s, print_special_tokens)] for s in samples]

    payload = {'predictions': predictions,
               'z': z,
               'c': c}
    return payload

