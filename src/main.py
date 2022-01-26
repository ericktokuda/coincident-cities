#!/usr/bin/env python3
"""Calculate the coincidence index
"""

import argparse
import random
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
def plot_weighted_graph(adjorig, labels, ethresh, layout, outpath):
    adj = np.tril(adjorig.copy()) # Lower triangle, to avoid double edges
    # print(np.min(adj))
    adj[adj < ethresh] = 0
    g = igraph.Graph(n=len(adjorig), directed=False)
    xx, yy = np.where(adj != 0)
    ewidths = []
    for x, y in zip(xx, yy):
        g.add_edge(x, y, w=adj[x, y])
        ewidths.append(adj[x, y])
    ewidths = np.array(ewidths)
    wmax = 3 # max edge widths
    # print(ewidths.shape)
    erange = np.max(ewidths) - np.min(ewidths)

    if erange == 0: # All edges have the same weight
        ewidths = np.ones(len(ewidths))
    else:
        ewidths = ( (ewidths - np.min(ewidths)) / erange ) * wmax

    igraph.plot(g, outpath, vertex_frame_width=1,
                vertex_frame_color='#dddddd', vertex_label=labels,
                # vertex_label_size=16, vertex_size=50, edge_color='#aaaaaa',
                vertex_label_size=12, vertex_size=30, edge_color='#aaaaaa',
                edge_width=list(ewidths), vertex_color='#809fff', margin=50,
                # edge_width=list(ewidths), vertex_color='#e27564', margin=50,
                layout=layout,
                )

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
def get_pearson_graph(dataorig, alpha, standardize):
    """Get graph of individual elements"""

    n, m = dataorig.shape
    if standardize:
        data = StandardScaler().fit_transform(dataorig)
    else:
        data = dataorig
    combs = list(combinations(range(n), 2))

    adj = np.zeros((n, n), dtype=float)

    import scipy
    for comb in combs:
        data2 = data[list(comb)]
        # c = coincidence(data2, alpha)
        c, _ = scipy.stats.pearsonr(data2[0, :], data2[1, :])
        adj[comb[0], comb[1]] = adj[comb[1], comb[0]] = c

    return adj

##########################################################
def main(csvpath, idcol, featcols, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    seed = 6
    random.seed(seed); np.random.seed(seed)
    alpha = .6
    edgethresh1 = .2
    edgethresh2 = .5

    df = pd.read_csv(csvpath)

    if idcol == None:
        rowids = [str(i) for i in range(len(df))]
    else:
        rowids = df[idcol]

    if featcols == None:
        data = df.to_numpy()
        feats = df.columns
    else:
        feats = featcols.split(',')
        data = df[feats].to_numpy()

    n, m = data.shape

    datameta = [] # Each row will correspond to a flattened adj matrix
    labels = [] # Label of each row of datameta

    # rowlabels = [rowid.capitalize() for rowid in df.city]
    rowlabels = [chr(i) for i in range(97, 97+n)]

    # m = 1
    for mm in range(1, m +1): # Size of the subset varies from 1 to m
        # combs = [list(range(12))]
        combs = list(combinations(range(m), mm))
        for comb in combs:
            combids = list(comb)
            suff = '_'.join([str(ind+1) for ind in combids])
            info('Combination ', suff)
            # suff = '_'.join([feats[ind] for ind in combids])
            adj = get_coincidence_graph(data[:, combids], alpha, True)
            np.savetxt(pjoin(outdir, '{}.txt'.format(suff)), adj)
            plotpath = pjoin(outdir, suff + '.pdf')
            plot_weighted_graph(adj, rowlabels, edgethresh1, 'fr', plotpath)
            datameta.append(adj.flatten())
            labels.append(suff)

    datameta = np.array(datameta)
    adj = get_coincidence_graph(datameta, alpha, False)
    vweights = np.sum(adj, axis=0)
    inds = np.argsort(np.sum(adj, axis=0))
    desc = list(reversed(inds))
    info('"Strongest" nodes: {}'.format(np.array(labels)[desc]))

    np.savetxt(pjoin(outdir, 'adj.txt'), adj)
    # adj = get_pearson_graph(datameta, alpha, False)
    plot_weighted_graph(adj, labels, edgethresh2, 'fr', pjoin(outdir, 'meta.pdf'))

    # vweights = ['{:.1f}'.format(w) for w in np.sum(adj, axis=0)]
    # plot_weighted_graph(adj, vweights, edgethresh2, pjoin(outdir, 'weights.pdf'))

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csvpath', default='data/data.csv', help='Input data')
    parser.add_argument('--idcol', default=None, help='Name of the label columns')
    parser.add_argument('--featcols', default=None, help='Subset of feats to consider, separated by comma')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.csvpath, args.idcol, args.featcols, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
