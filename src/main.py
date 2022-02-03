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
from myutils.transform import pca, get_pc_contribution
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import igraph
import seaborn as sns

VCLR1 = '#809fff'
VCLR2 = '#75bb79'
VFCLR = '#dddddd'
ECLR1  = '#aaaaaaff'
ECLR2  = '#55555599'

##########################################################
def plot_feats_pca(df, outdir):
    """Principal component analysis of the features"""
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, 'pca.png')
    if os.path.exists(outpath): return
    featlabels = df.columns.tolist()
    n, m = df.shape
    tr, evecs, evals = pca(df.to_numpy(), normalize=True)
    pcs, contribs, relcontribs = get_pc_contribution(evecs, evals)
    info('Contribs:', np.array(featlabels)[pcs], contribs, relcontribs)

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    ax.scatter(tr[:, 0], tr[:, 1], s=380, c=VCLR1, edgecolors=ECLR1)
    # for i in range(tr.shape[0]):
        # ax.scatter(tr[i, 0], tr[i, 1], label=cities[i])

    for pos in ['right', 'top']: ax.spines[pos].set_visible(False)

    rowlabels = []
    for i in range(n):
        rowlabels.append('{} ({})'.format(df.index[i].capitalize(), chr(i + 97)))

    for i in range(n):
        ax.annotate(rowlabels[i], (tr[i, 0], tr[i, 1]), ha='center', va='center')

    xylim = np.max(np.abs(tr[:, 0:2])) * 1.1
    # ax.set_xlabel('PCA1 ({} ({}%)'.format(cols[pcs[0]], int(contribs[0] * 100)))
    # ax.set_ylabel('PCA2 {}: ({}%)'.format(cols[pcs[1]], int(contribs[1] * 100)))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.tight_layout()
    plt.savefig(outpath)

##########################################################
def plot_feats_correlogram(df, outdir):
    """Plot pairwise correlation of the features"""
    info(inspect.stack()[0][3] + '()')
    plt.style.use('seaborn')
    outpath = pjoin(outdir, 'pairwise_correl.png')
    if os.path.exists(outpath): return

    f = 1.5
    figsize = np.array([1, .75]) * f # W, H
    corrs = df.corr()
    sns.pairplot(corrs, corner=True, height=figsize[1], aspect=figsize[0]/figsize[1])
    plt.tight_layout()
    plt.savefig(outpath)
    plt.style.use('default')

##########################################################
def plot_histograms(outdir):
    """Plot histograms based on the adhocly definedclusters"""
    info(inspect.stack()[0][3] + '()')

    n = 5
    clusters = [
        [[3,5], [1,3,5], [2,3,5], [3,4,5], [1,3,4,5], [1,2,3,5], [2,3,4,5], [1,2,3,4,5]],
        [[1,3,4], [3,4], [1,2,3,4], [2,3,4], [1,2,3], [2,3], [1,3]],
        [[5], [4, 5], [2, 4, 5], [1,4,5], [1,2,5], [2,5], [1,2,4,5], [1,5]],
        [[1,2,4], [1,4], [1,2], [2,4], [4]]
    ]

    for clid, cluster in enumerate(clusters):
        counter = np.zeros(n, dtype=float)
        clsizes = []
        for i, nodes in enumerate(cluster):
            clsizes.append(len(nodes))
            for node in nodes:
                counter[node - 1] += 1

        W = 320; H = 240

        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        # ax.bar(range(1, n+1), np.array(counter) / np.sum(counter))
        ax.bar(range(1, n+1), np.array(counter), color='#66BB6A')
        ax.set_xlabel('Feature id')
        ax.set_ylim(0, 9)
        plt.tight_layout()

        outpath = pjoin(outdir, 'meta_hist{}.png'.format(clid))
        plt.savefig(outpath)

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

def test_read_igraph():
    adj = np.arange(.1, 1, .1).astype(float).reshape(3, 3)

##########################################################
def get_igraphobj_from_adj(adjorig, ethresh=0):
    """Convert real-valued adjacency matrix to an igraph weighted instance. """
    adj = np.tril(adjorig.copy(), k=-1) # Lower tri without maindiag, to avoid double edges
    adj[adj < ethresh] = 0
    return igraph.Graph.Weighted_Adjacency(adj, mode='undirected')

##########################################################
def plot_weighted_graph(adj, labels, ethresh, plotargs, outpath):
    g = get_igraphobj_from_adj(adj, ethresh)
    ewidths = np.array(g.es['weight'])
    wmax = 3 # max edge widths
    erange = np.max(ewidths) - np.min(ewidths)

    if erange == 0: # All edges have the same weight
        ewidths = np.ones(len(ewidths))
    else:
        ewidths = ( (ewidths - np.min(ewidths)) / erange ) * wmax

    # plt.style.use('ggplot')
    # fig, ax = plt.subplots(figsize=(16, 16))
    # fig, ax = plt.subplots(figsize=(4, 4))
    # igraph.plot(g, target=ax, edge_width=.5, layout=layout)
    igraph.plot(g, outpath, edge_width=ewidths, **plotargs)

    # plt.savefig(outpath); plt.close()
    # igraph.plot(g, outpath, bbox=(1200, 1200))
    return
    igraph.plot(g, outpath, vertex_frame_width=1,
                vertex_frame_color='#dddddd', vertex_label=labels,
                # vertex_label_size=16, vertex_size=50, edge_color='#aaaaaa',
                vertex_label_size=12, vertex_size=30, edge_color='#aaaaaa',
                edge_width=list(ewidths), vertex_color='#75bb79', margin=50,
                # edge_width=list(ewidths), vertex_color='#e27564', margin=50,
                layout=layout,
                )

##########################################################
def get_coincidx_graph(dataorig, alpha, standardize):
    """Get graph of individual elements"""
    # info(inspect.stack()[0][3] + '()')

    n, m = dataorig.shape
    if standardize:
        data = StandardScaler().fit_transform(dataorig)
    else:
        data = dataorig

    adj = np.zeros((n, n), dtype=float)
    for comb in list(combinations(range(n), 2)):
        data2 = data[list(comb)]
        c = coincidence(data2, alpha)
        adj[comb[0], comb[1]] = adj[comb[1], comb[0]] = c

    return adj

##########################################################
def get_pearson_graph(dataorig, standardize):
    """Get graph of individual elements"""
    info(inspect.stack()[0][3] + '()')

    n, m = dataorig.shape
    if standardize:
        data = StandardScaler().fit_transform(dataorig)
    else:
        data = dataorig

    adj = np.zeros((n, n), dtype=float)
    for comb in list(combinations(range(n), 2)):
        data2 = data[list(comb)]
        c, _ = scipy.stats.pearsonr(data2[0, :], data2[1, :])
        adj[comb[0], comb[1]] = adj[comb[1], comb[0]] = c

    return adj

##########################################################
def plot_heatmaps(adjdir, outdir):
    info(inspect.stack()[0][3] + '()')

    os.makedirs(outdir, exist_ok=True)
    labels = [chr(i) for i in range(97, 97 + 21)]
    for f in sorted(os.listdir(adjdir)):
        if not f.endswith('.txt'): continue
        if 'README' in  f: continue
        # info(f)
        adj = np.loadtxt(pjoin(adjdir, f))
        fig, ax = plt.subplots()
        mask = np.triu(np.ones_like(adj, dtype=bool))
        sns.heatmap(adj, mask=mask, ax=ax, vmin=-1, vmax=+1, xticklabels=labels,
                    yticklabels=labels)
        plt.tight_layout()
        plt.savefig(pjoin(outdir, f.replace('.txt', '.png')))
        plt.close()

##########################################################
def get_coincidx_of_feats(df, alpha, edgethresh1, outrootdir):
    """Calculate coincidence index of the rows"""
    info(inspect.stack()[0][3] + '()')
    data = df.to_numpy()
    n, m = data.shape

    outdir = pjoin(outrootdir, 'combs')
    coirowsflatcsv = pjoin(outrootdir, 'coirowsflat.csv')
    if os.path.exists(coirowsflatcsv):
        return pd.read_csv(coirowsflatcsv, index_col=0)

    os.makedirs(outdir, exist_ok=True)
    adjs = []
    labels = [] # Label of each row of datameta

    # rowlabels = [chr(i) for i in range(97, 97+n)]
    rowlabels = df.index.tolist()

    layout = dict(vertex_frame_width=1,
                    vertex_frame_color=VFCLR,
                    vertex_label=rowlabels,
                    # vertex_label_size=16, vertex_size=50, edge_color='#aaaaaa',
                    vertex_label_size=12, vertex_size=50, edge_color=ECLR1,
                    vertex_color=VCLR1, margin=50,
                    # edge_width=list(ewidths), vertex_color='#e27564', margin=50,
                  #75bb79
                    # layout='circle',
                    layout='fr',
                    )

    for mm in range(1, m +1): # Size of the subset varies from 1 to m
        combs = list(combinations(range(m), mm))
        info('Combinations of size ', mm)
        for comb in combs:
            combids = list(comb)
            suff = '_'.join([str(ind+1) for ind in combids])
            adj = get_coincidx_graph(data[:, combids], alpha, True)
            np.savetxt(pjoin(outdir, '{}.txt'.format(suff)), adj)
            plotpath = pjoin(outdir, suff + '.png')
            plot_weighted_graph(adj, rowlabels, edgethresh1, layout, plotpath)
            adjs.append(adj)
            labels.append(suff)

    msk = np.tril(np.ones((n, n))).astype(bool) # Just lower triangle
    coirowsflat = np.array([matrx[msk] for matrx in adjs]) # Flatten matrices
    dfcombs = pd.DataFrame(coirowsflat, index=labels)
    dfcombs.to_csv(coirowsflatcsv)
    return dfcombs

# ##########################################################
# def get_coincidx_of_rows11(df, alpha, edgethresh1, outrootdir):
#     """Calculate coincidence index of the rows"""
#     info(inspect.stack()[0][3] + '()')
#     data = df.to_numpy()
# 
#     n, m = data.shape
# 
#     outdir = pjoin(outrootdir, 'rows11')
#     os.makedirs(outdir, exist_ok=True)
#     adjs = []
#     labels = [] # Label of each row of datameta
# 
#     rowlabels = [chr(i) for i in range(97, 97+n)]
# 
#     m = 1
#     for mm in range(1, m +1): # Size of the subset varies from 1 to m
#         # combs = [list(range(11))]
#         combs = [list(range(9))]
#         # combs = list(combinations(range(m), mm))
#         for comb in combs:
#             combids = list(comb)
#             suff = '_'.join([str(ind+1) for ind in combids])
#             info('Combination ', suff)
#             # suff = '_'.join([feats[ind] for ind in combids])
#             adj = get_coincidx_graph(data[:, combids], alpha, True)
#             np.savetxt(pjoin(outdir, '{}.txt'.format(suff)), adj)
#             plotpath = pjoin(outdir, suff + '.pdf')
#             plot_weighted_graph(adj, rowlabels, edgethresh1, 'fr', '#809fff', plotpath)
#             adjs.append(adj)
#             labels.append(suff)
#     return adjs, labels

##########################################################
def get_pearson_of_coincidx(dfcoincidx, outrootdir):
    outdir = pjoin(outrootdir, 'pearson')
    outpath = pjoin(outdir, 'pearson.png')
    if os.path.exists(outpath): return
    os.makedirs(outdir, exist_ok=True)

    rowlabels = dfcoincidx.index.tolist()
    adj = get_pearson_graph(dfcoincidx.to_numpy(), False)
    plot_weighted_graph(adj, rowlabels, 0.5, 'fr', '#75bb79', outpath)

##########################################################
def plot_communities(adj, rowlabels, edgethresh, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    g = get_igraphobj_from_adj(adj, edgethresh)
    # breakpoint()
    layout = g.layout('fr', weights='weight')
    # layout = g.layout('kk')

    # Plot the labels in vectorial format
    plotargs = dict(bbox=(1200, 1200),
                    vertex_frame_width=.7,
                    vertex_frame_color=VFCLR,
                    vertex_color=VCLR2,
                    vertex_size=8, edge_color=ECLR2,
                    vertex_label=rowlabels,
                    vertex_label_size=5,
                    margin=50,
                    edge_width=0.4,
                    layout=layout,
                    )
    igraph.plot(g, pjoin(outdir, 'labels.pdf'), **plotargs)

    # Plot different community detection approaches
    plotargs = dict(bbox=(1000, 1000),
                    vertex_frame_width=.7,
                    vertex_frame_color=VFCLR,
                    vertex_size=10, edge_color=ECLR2,
                    margin=50,
                    edge_width=0.4,
                    layout=layout,
                    )

    # info('infomap')
    # comms = g.community_infomap(edge_weights='weight')
    # igraph.plot(comms, pjoin(outdir, 'infomap.png'), **plotargs)

    # info('labelprop')
    # comms = g.community_label_propagation(weights='weight')
    # igraph.plot(comms, pjoin(outdir, 'labelprop.png'), **plotargs)


    # # info('spinglass')
    # # comms = g.community_spinglass(weights='weight')
    # # igraph.plot(comms, pjoin(outdir, 'spinglass.png'), **plotargs)

    # # info('edgebetw')
    # # comms = g.community_edge_betweenness(directed=False, weights='weight')
    # # igraph.plot(comms, pjoin(outdir, 'edgebetw.png'), **plotargs)

    info('multilevel')
    comms = g.community_multilevel(weights='weight', return_levels=False)
    igraph.plot(comms, pjoin(outdir, 'multilevel.png'), **plotargs)

    g2 = comms.cluster_graph()
    igraph.plot(g2, pjoin(outdir, 'multilevel_clgraph.png'))

    return layout

##########################################################
def get_coincidx_of_coincidx(df, alpha, edgethresh2, outdir):
    info(inspect.stack()[0][3] + '()')
    adj = get_coincidx_graph(df.to_numpy(), alpha, False)
    rowlabels = df.index.to_list()

    # layout = plot_communities(adj, rowlabels, edgethresh2, outdir)

    # vweights = np.sum(adj, axis=0)
    # inds = np.argsort(np.sum(adj, axis=0))
    # desc = list(reversed(inds))
    # df = pd.DataFrame(np.array([rowlabels, vweights]).T, columns=['comb', 'weight'])
    # df.to_csv(pjoin(outdir, 'weights.csv'), index=False)

    np.savetxt(pjoin(outdir, 'meta.txt'), adj)

    # return

    outpath = pjoin(outdir, 'meta.png')
    plotargs = dict(bbox=(500, 500),
                    vertex_color=VCLR2, margin=50,
                    vertex_frame_width=.5,
                    vertex_frame_color=VFCLR,
                    vertex_size=10, edge_color=ECLR2,
                    layout='fr',
                    )

    layout = plot_weighted_graph(adj, rowlabels, edgethresh2, plotargs, outpath)

    # vweights = ['{:.1f}'.format(w) for w in np.sum(adj, axis=0)]
    # plot_weighted_graph(adj, vweights, edgethresh2, layout, '#66BB6A', pjoin(outdir, 'weights.pdf'))

##########################################################
def main(csvpath, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    seed = 1
    random.seed(seed); np.random.seed(seed)

    featsoutdir = pjoin(outdir, 'feats')
    coincidxoutdir = pjoin(outdir, 'coincidx')
    os.makedirs(featsoutdir, exist_ok = True)
    os.makedirs(coincidxoutdir, exist_ok = True)

    ethreshfeats   = .2
    ethreshcoinc   = .55
    ethreshpearson = .8
    alpha = .6

    dforig = pd.read_csv(csvpath)
    rowids = dforig['city'].tolist()
    # cols05 = ['degstd', 'transstd', 'vposstd2', 'degmean', 'acc05std']
    cols05 = ['degstd', 'transstd', 'vposstd2', 'eangstd', 'acc05std']
    # cols11 = [ 'degmean', 'degstd', 'deg3', 'deg4', 'deg5', 'transmean', 'transstd',
              # 'eangstd', 'vposstd2', 'acc05mean', 'acc05std', 'lacun21']
    # dffeats = dforig.set_index('city')[cols11]
    dffeats = dforig.set_index('city')[cols05]

    plot_feats_correlogram(dffeats, featsoutdir)
    plot_feats_pca(dffeats, featsoutdir)

    dfcoincidx = get_coincidx_of_feats(dffeats, alpha, ethreshfeats, featsoutdir)
    # plot_heatmaps(outdir, pjoin(outdir, 'heatmaps/'))
    get_coincidx_of_coincidx(dfcoincidx, alpha, ethreshcoinc, coincidxoutdir)

    # dfpearson = get_pearson_of_feats(dffeats, coincidxoutdir)
    # get_pearson_of_pearson(dfpearson, ethreshpearson, coincidxoutdir)
    # plot_histograms(outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csvpath', default='data/data.csv', help='Input data')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.csvpath, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
