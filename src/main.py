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
EXT = '.pdf'
 # teal, orange, blue, pink, green
PALETTE1 = np.array([ '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', ])
PALETTE2 = np.array([
        '#66c2a5','#66c2a5','#66c2a5','#66c2a5',
        '#fc8d62','#fc8d62','#fc8d62','#fc8d62',
        '#8da0cb','#8da0cb','#8da0cb','#8da0cb',
        '#e78ac3','#e78ac3','#e78ac3','#e78ac3',
        '#a6d854','#a6d854','#a6d854','#a6d854',
        ])
PALETTE3 = ['#bdd3cf','#d9d9d9','#fccde5','#ffff9d','#80b1d3',
        '#fdb462','#b3de69','#bebada','#bda6fb']
countries = ['France', 'Germany', 'Italy', 'Spain', 'U.K.']

##########################################################
def plot_feats_pca(df, outdir):
    """Principal component analysis of the features"""
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, 'pca' + EXT)
    featlabels = df.columns.tolist()
    n, m = df.shape
    tr, evecs, evals = pca(df.to_numpy(), normalize=True)
    pcs, contribs, relcontribs = get_pc_contribution(evecs, evals)
    info('Contribution of each component:{}', contribs)
    info('Most important variable in each component: ', np.array(featlabels)[pcs], relcontribs)

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    for i in range(len(countries)):
        ax.scatter(tr[i*4:(i+1)*4, 0], tr[i*4:(i+1)*4, 1],
                s=380, c=PALETTE1[i], edgecolors=ECLR1, label=countries[i])
    ax.legend(loc='lower right', markerscale=.5)

    for pos in ['right', 'top']: ax.spines[pos].set_visible(False)

    rowlabels = []
    for i in range(n):
        rowlabels.append('{}'.format(df.index[i].capitalize()))

    for i in range(n):
        ax.annotate(rowlabels[i], (tr[i, 0], tr[i, 1]),
                    ha='center', va='center', fontsize='small')

    xylim = np.max(np.abs(tr[:, 0:2])) * 1.1
    ax.set_xlabel('PC 1 ({}%)'.format(int(contribs[0]*100)))
    ax.set_ylabel('PC 2 ({}%)'.format(int(contribs[1]*100)))
    plt.tight_layout()
    plt.savefig(outpath)

##########################################################
def plot_feats_correlogram(df, outdir):
    """Plot pairwise correlation of the features"""
    info(inspect.stack()[0][3] + '()')
    plt.style.use('seaborn')
    outpath = pjoin(outdir, 'pairwise_correl' + EXT)

    f = 1.5
    figsize = np.array([1, .75]) * f # W, H
    corrs = df.corr()
    sns.pairplot(corrs, corner=True, height=figsize[1], aspect=figsize[0]/figsize[1])
    plt.tight_layout()
    plt.savefig(outpath)
    plt.style.use('default')

##########################################################
def plot_histograms(comms, rowlabels, outdir):
    """Plot histogram of the elements in the vertex labels"""
    info(inspect.stack()[0][3] + '()')
    for clid, comm in enumerate(comms):
        commlbs = rowlabels[comm]
        m = len(commlbs)
        counts = np.zeros(5, dtype=int)
        for lbl in commlbs:
            featids = [int(ii) for ii in lbl.split('_')]
            for id_ in featids:
                counts[id_- 1] += 1

        outpath = pjoin(outdir, 'coinc_hist{}{}'.format(clid, EXT))
        plot_histogram(counts, outpath)

##########################################################
def plot_histogram(counts, outpath):
    """Plot histograms based on the adhocly definedclusters"""
    W = 200; H = 175
    n = len(counts)

    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.bar(range(1, n+1), np.array(counts), color='#66BB6A')
    ax.set_xlabel('Feature id')
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 9)
    plt.tight_layout()
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
def plot_weighted_graph(adj, ethresh, plotargs, outpath):
    g = get_igraphobj_from_adj(adj, ethresh)
    ewidths = np.array(g.es['weight'])
    wmax = 4 # max edge widths
    erange = np.max(ewidths) - np.min(ewidths)

    if erange == 0: # All edges have the same weight
        ewidths = np.ones(len(ewidths))
    else:
        ewidths = ( (ewidths - np.min(ewidths)) / erange ) * (wmax - 1) + 1

    igraph.plot(g, outpath, edge_width=ewidths, **plotargs)

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
        plt.savefig(pjoin(outdir, f.replace('.txt', EXT)))
        plt.close()

##########################################################
def get_coincidx_of_feats(df, alpha, edgethresh1, outrootdir):
    """Calculate coincidence index of the rows"""
    info(inspect.stack()[0][3] + '()')
    data = df.to_numpy()
    n, m = data.shape

    outdir = pjoin(outrootdir, 'combs')
    coirowsflatcsv = pjoin(outrootdir, 'coirowsflat.csv')

    os.makedirs(outdir, exist_ok=True)
    adjs = []
    labels = [] # Label of each row of datameta

    rowlabels = df.index.tolist()

    layout = dict(bbox=(400, 400), layout='fr', margin=50,
                  vertex_size=20, vertex_color=PALETTE2,
                  vertex_frame_width=1, vertex_frame_color=VFCLR,
                  vertex_label_size=12,
                  vertex_label=rowlabels,
                  edge_color=ECLR1
                  )

    for mm in range(1, m +1): # Size of the subset varies from 1 to m
        combs = list(combinations(range(m), mm))
        info('Combinations of size ', mm)
        for comb in combs:
            combids = list(comb)
            suff = '_'.join([str(ind+1) for ind in combids])
            adj = get_coincidx_graph(data[:, combids], alpha, True)
            np.savetxt(pjoin(outdir, '{}.txt'.format(suff)), adj)
            plotpath = pjoin(outdir, suff + EXT)
            plot_weighted_graph(adj, edgethresh1, layout, plotpath)

            g = get_igraphobj_from_adj(adj, edgethresh1)
            membership, _ = np.mgrid[:4, :5] # TODO: should adjust automatically
            modul = g.modularity(membership.flatten(), weights='weight')
            if mm == m:
                open(pjoin(outrootdir, 'modularity.txt'), 'a'). \
                    write('{},{}\n'.format(alpha, modul))
                # We can then cat them all (in bash)

            adjs.append(adj)
            labels.append(suff)

    msk = np.tril(np.ones((n, n))).astype(bool) # Just lower triangle
    coirowsflat = np.array([matrx[msk] for matrx in adjs]) # Flatten matrices
    dfcombs = pd.DataFrame(coirowsflat, index=labels)
    dfcombs.to_csv(coirowsflatcsv)
    return dfcombs

##########################################################
def get_pearson_of_coincidx(dfcoincidx, outrootdir):
    outdir = pjoin(outrootdir, 'pearson')
    outpath = pjoin(outdir, 'pearson' + EXT)
    os.makedirs(outdir, exist_ok=True)

    rowlabels = dfcoincidx.index.tolist()
    adj = get_pearson_graph(dfcoincidx.to_numpy(), False)
    plot_weighted_graph(adj, 0.5, 'fr', '#75bb79', outpath)

##########################################################
def plot_communities(adj, edgethresh, plotargs, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    g = get_igraphobj_from_adj(adj, edgethresh)
    rowlabels = np.array(plotargs['vertex_label'])

    info('Multilevel method') # It could be infomap, label_propagation, edge_betweenness
    comms = g.community_multilevel(weights='weight', return_levels=False)

    info('Modularity: {}'.format(g.modularity(comms)))

    colours = np.empty(g.vcount(), dtype=object) # Assign colour according to the comm.
    for i, comm in enumerate(list(comms)):
        colours[comm] = PALETTE3[i]

    plotargs['vertex_color'] = colours
    plot_weighted_graph(adj > edgethresh, edgethresh, plotargs,
                        pjoin(outdir, 'multilevel' + EXT))
    del plotargs['vertex_color']

    plot_histograms(comms, rowlabels, outdir)
    # vweights = np.sum(adj, axis=0)
    # Distribution of features per community

    
    # g2 = comms.cluster_graph() # Plot cluster graph
    # igraph.plot(g2, pjoin(outdir, 'multilevel_clgraph' + EXT))

##########################################################
def get_coincidx_of_coincidx(df, alpha, edgethresh2, outdir):
    info(inspect.stack()[0][3] + '()')
    adj = get_coincidx_graph(df.to_numpy(), alpha, False)

    np.savetxt(pjoin(outdir, 'coincofcoinc.txt'), adj)

    outpath = pjoin(outdir, 'coincofcoinc' + EXT)

    g = get_igraphobj_from_adj(adj, edgethresh2)

    plotargs = dict(bbox=(500, 500), margin=50,
                  vertex_size=20, vertex_color=VCLR2,
                  vertex_frame_width=1, vertex_frame_color=VFCLR,
                  vertex_label=df.index.to_list(), vertex_label_size=9,
                  edge_color=ECLR1, layout=g.layout('fr', weights='weight')
                  )

    plot_weighted_graph(adj, edgethresh2, plotargs.copy(), outpath)

    del plotargs['vertex_color']
    plot_communities(adj, edgethresh2, plotargs.copy(), outdir)

    vweights = ['{:.1f}'.format(w) for w in np.sum(adj, axis=0)]
    plotargs['vertex_label'] = vweights
    plot_weighted_graph(adj, edgethresh2, plotargs, pjoin(outdir, 'weights.pdf'))

##########################################################
def plot_feats_eventplot(df, outdir):
    """Plot features according to """
    W = 320; H = 240
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    groups = [
            'Bradford,Southampton,Derby,Luton,Vigo'.split(','),
            'Brunswick,Kiel,Verona,Freiburg,Messina'.split(','),
            'Rennes,Lille,Bordeaus,Granda'.split(','),
            'Bari,Oviedo,Padova'.split(','),
            ]
    
    data = df.loc[groups[1]].values.T
    
    import sklearn
    from sklearn.preprocessing import normalize, MinMaxScaler
    # data = normalize(data, axis=0)
    data = MinMaxScaler().fit_transform(data.T).T
    
    ax.eventplot(data, linelengths=.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks(range(5))
    # ax.set_yticklabels(df.columns)
    ax.set_yticklabels('degavg,degstd,transstd,vposdisp,accessib'.split(','))
    plt.tick_params(left=False, bottom=False)
    for i in range(5):
        plt.hlines(i,0,1)  # Draw a horizontal line
    # axs.legend(labels, bbox_to_anchor=(0., 1.0, 1., .10), loc=3,ncol=3, mode="expand", borderaxespad=0.)
    ax.annotate('0', (0, -.7),
                ha='center', va='center', fontsize='small')
    ax.annotate('1.0', (1, -.7),
                ha='center', va='center', fontsize='small')
    # ax.set_xlabel('Normalized value')
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'feats1d.pdf'))

##########################################################
def plot_feats_radarplot(df, countries, outdir):
    """Plot features according to """
    W = 320; H = 240
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    groups = [
            'Bradford,Southampton,Derby,Luton,Vigo'.split(','),
            'Brunswick,Kiel,Verona,Freiburg,Messina'.split(','),
            'Rennes,Lille,Bordeaux,Granada'.split(','),
            'Bari,Oviedo,Padova'.split(','),
            ]
    
    cities = df.index.values
    grcountries = []
    for i, gr in enumerate(groups):
        ids = []
        for el in groups[i]:
            x = np.where(cities == el)[0][0]
            ids.append(x)
        grcountries.append(ids)

    for Z in range(4):
        data = df.loc[groups[Z]].values

        import sklearn
        from sklearn.preprocessing import normalize, MinMaxScaler
        data = MinMaxScaler().fit_transform(data)
        
        label_loc = np.linspace(0, 2 * np.pi, 6)

        plt.figure(figsize=(4, 4))
        ax = plt.subplot(polar=True)

        for i in range(data.shape[0]):
            data2 = [*data[i, :], data[i, 0]]
            ax.plot(label_loc, data2, c=PALETTE2[grcountries[Z][i]])
        
        # ax.set_frame_on(False)
        ax.yaxis.grid(False)
        ax.set_yticks([])
        # ax.yaxis.set_visible(False)
        ax.set_rorigin(-.1)
        ax.spines['polar'].set_visible(False)

        cols = 'degavg,degstd,transstd,vposdisp,accessib'.split(',')
        cols = [*cols, cols[0]]
        lines, labels = plt.thetagrids(np.degrees(label_loc), labels=cols)

        plt.tight_layout()
        plt.savefig(pjoin(outdir, 'comm{}_radar.pdf'.format(Z)))
        plt.close()

##########################################################
def main(csvpath, alphafeats, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    seed = 1
    random.seed(seed); np.random.seed(seed)

    featsoutdir = pjoin(outdir, 'feats')
    coincidxoutdir = pjoin(outdir, 'coincidx')
    os.makedirs(featsoutdir, exist_ok = True)
    os.makedirs(coincidxoutdir, exist_ok = True)

    ethreshfeats   = .1
    ethreshcoinc   = .55
    # ethreshpearson = .8
    # alphafeats = .35
    alphacoinc = .6

    dforig = pd.read_csv(csvpath)
    rowids = dforig['city'].tolist()
    cols = ['degmean', 'degstd', 'transstd', 'vposstd2', 'acc05std']
    # cols = [ 'degmean', 'degstd', 'deg3', 'transmean', 'transstd',
              # 'eangstd', 'vposstd2', 'acc05mean', 'acc05std']
    dffeats = dforig.set_index('city')[cols]
    countries = dforig.country.tolist()

    # plot_feats_eventplot(dffeats, outdir)
    # plot_feats_radarplot(dffeats, countries, outdir)
    # plot_feats_correlogram(dffeats, featsoutdir)
    plot_feats_pca(dffeats, featsoutdir)
    dfcoincidx = get_coincidx_of_feats(dffeats, alphafeats, ethreshfeats, featsoutdir)

    # plot_heatmaps(outdir, pjoin(outdir, 'heatmaps/'))
    adj = get_coincidx_of_coincidx(dfcoincidx, alphacoinc, ethreshcoinc, coincidxoutdir)

    # dfpearson = get_pearson_of_feats(dffeats, coincidxoutdir)
    # get_pearson_of_pearson(dfpearson, ethreshpearson, coincidxoutdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alphaf', type=float, default=0.288889,
                        help='Alpha of the coincidence index')
    parser.add_argument('--csvpath', default='data/data.csv', help='Input data')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.csvpath, args.alphaf, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
