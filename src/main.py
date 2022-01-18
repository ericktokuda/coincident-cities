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
import igraph

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
    return inter * jac

##########################################################
def plot_weighted_graph(adjorig, labels, ethresh, outpath):
    adj = np.tril(adjorig.copy()) # Lower triangle, to avoid double edges
    adj[adj < ethresh] = 0
    g = igraph.Graph(n=len(adjorig), directed=False)
    xx, yy = np.where(adj != 0)
    ewidths = []
    for x, y in zip(xx, yy):
        g.add_edge(x, y, w=adj[x, y])
        ewidths.append(adj[x, y])
    ewidths = np.array(ewidths)
    wmax = 4 # max edge widths
    erange = np.max(ewidths) - np.min(ewidths)

    if erange == 0: # All edges have the same weight
        ewidths = np.ones(len(ewidths))
    else:
        ewidths = ( (ewidths - np.min(ewidths)) / erange ) * wmax

    igraph.plot(g, outpath, vertex_frame_width=0, vertex_label=labels, edge_width=ewidths,
                                            vertex_color='yellow')

##########################################################
def get_coincidence_graph(dataorig, alpha, standardize):
    """Get graph of individual elements"""

    n, m = dataorig.shape
    if standardize:
        data = StandardScaler().fit_transform(dataorig)
    else:
        data = dataorig
    combs = list(combinations(range(n), 2))

    adj = np.zeros((n, n), dtype=float)

    for comb in combs:
        data2 = data[list(comb)]
        c = coincidence(data2, alpha)
        adj[comb[0], comb[1]] = adj[comb[1], comb[0]] = c

    return adj

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')
    csvpath = 'data/particles.csv'
    feats = ['spin', 'charge', 'mass']
    alpha = .4
    edgethresh = .35

    df = pd.read_csv(csvpath)
    data = df[feats].to_numpy()
    n, m = data.shape

    datameta = [] # Each row will correspond to a flattened adj matrix
    labels = [] # Label of each row of datameta

    for mm in range(1, m +1): # Size of the subset varies from 1 to m
        combs = list(combinations(range(m), mm))
        for comb in combs:
            combids = list(comb)
            suff = '_'.join([str(ind) for ind in combids])
            adj = get_coincidence_graph(data[:, combids], alpha, True)
            plotpath = pjoin(outdir, suff + '.png')
            vstr = [str(ii) for ii in range(n)]
            plot_weighted_graph(adj, vstr, edgethresh, plotpath)
            datameta.append(adj.flatten())
            labels.append(suff)

    datameta = np.array(datameta)
    adj = get_coincidence_graph(datameta, alpha, False)
    plot_weighted_graph(adj, labels, edgethresh, pjoin(outdir, 'meta.png'))

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
