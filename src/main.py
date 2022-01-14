#!/usr/bin/env python3
"""Calculate the coincidence index
"""

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations

##########################################################
def interiority(dataorig):
    """Calculate the interiority index of the two rows. @vs has 2rows and n-columns, where
    n is the number of features"""
    # info(inspect.stack()[0][3] + '()')
    data = np.abs(dataorig)
    abssum = np.sum(data, axis=1)
    den = np.min(abssum)
    num = np.sum(np.min(data, axis=0))
    return num / den
    
##########################################################
def jaccard(dataorig, a):
    """Calculate the interiority index of the two rows. @vs has 2rows and n-columns, where
    n is the number of features"""
    # info(inspect.stack()[0][3] + '()')
    data = np.abs(dataorig)
    den = np.sum(np.max(data, axis=0))
    datasign = np.sign(dataorig)
    plus_ = np.abs(datasign[0, :] + datasign[1, :])
    minus_ = np.abs(datasign[0, :] - datasign[1, :])
    splus = np.sum(plus_ * np.min(data, axis=0))
    sminus = np.sum(minus_ * np.min(data, axis=0))
    num = a * splus - (1 - a) * sminus
    return num / den
    

##########################################################
def coincidence(data, a):
    inter = interiority(data)
    jac = jaccard(data, a)
    print(inter, jac)
    return inter * jac

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')
    csvpath = 'data/particles.csv'
    feats = ['spin', 'charge', 'mass']
    # feats = ['spin', 'charge']
    df = pd.read_csv(csvpath)
    n, m = len(df), len(feats)
    a = .6
    normalized = StandardScaler().fit_transform(df[feats])

    combs = list(combinations(range(n), 2))
    
    import igraph
    g = igraph.Graph(n=18)

    acc = 0
    for comb in combs:
        data = normalized[list(comb)]
        c = coincidence(data, a)
        if c > .35:
            print(comb, c)
            acc += 1
            g.add_edge(comb[0], comb[1])
    igraph.plot(g, '/tmp/out.png')

    print(acc)
    
##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
