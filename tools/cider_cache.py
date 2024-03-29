import os
import sys
import json
import h5py
import numpy as np
import argparse
import pickle
from collections import defaultdict
import ipdb

def precook(words, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

def remove_ignore(seq):
    sent = []
    for word in seq:
        if word == -1:
            break
        sent.append(word)
    return sent

def main(args):
    gts = []
    with open(args.image_ids) as f:
        image_ids = [line.strip() for line in f]

    for _ in range(len(image_ids)):
        gts.append([])

    target_seqs = pickle.load(open(args.infile, 'rb'), encoding='bytes')
    for i, image_id in enumerate(image_ids):
        seq = np.array(target_seqs[image_id]).astype('int')
        #ipdb.set_trace()
        # for seq in seqs:
        gts[i].append(remove_ignore(seq))
    pickle.dump(gts, open(args.gts, 'wb'))

    crefs = []
    for gt in gts:
        crefs.append(cook_refs(gt))

    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
            document_frequency[ngram] += 1
    ref_len = np.log(float(len(crefs)))
    pickle.dump({ 'document_frequency': document_frequency, 'ref_len': ref_len }, open(args.outfile, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input h5
    parser.add_argument('--infile', default='./mscoco/sent/mimic_train_target.pkl', help='pkl file', type=str)
    parser.add_argument('--outfile', default='./mscoco/misc/mimic_train_cider.pkl', help='output pickle file', type=str)
    parser.add_argument('--gts', default='./mscoco/misc/mimic_train_gts.pkl', help='output pickle file', type=str)
    parser.add_argument('--image_ids', default='./mscoco/txt/mimic_train_image_id_new2.txt', help='image id file', type=str)

    args = parser.parse_args()
    main(args)
