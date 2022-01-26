#!/usr/bin/env python3
"""Plot results after running main.py
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
plt.style.use('ggplot')
import pandas as pd
from myutils import info, create_readme
from myutils.transform import pca, get_pc_contribution

##########################################################
def plot_pca(df, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    cols = [ 'degmean', 'degstd', 'deg3', 'deg4', 'deg5', 'transmean', 'transstd', 'eangstd',
    'vposstd2', 'lacun21', 'acc05mean', 'acc05std']

    df2 = df[cols].copy(deep=True)
    data = df2.to_numpy()
    tr, evecs, evals = pca(data, normalize=True)
    pcs, contribs, relcontribs = get_pc_contribution(evecs, evals)
    info('Contribs:', np.array(cols)[pcs], contribs, relcontribs)

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    # ax.set_title('PCA components (pc1 and pc2)')

    cities = [ c.capitalize() for c in list(df.city)]

    ax.scatter(tr[:, 0], tr[:, 1])
    # for i in range(tr.shape[0]):
        # ax.scatter(tr[i, 0], tr[i, 1], label=cities[i])

    for i in range(len(df2)):
        ax.annotate(df.iloc[i].city.capitalize(), (tr[i, 0], tr[i, 1]))

    xylim = np.max(np.abs(tr[:, 0:2])) * 1.1
    # ax.set_xlabel('PCA1 ({} ({}%)'.format(cols[pcs[0]], int(contribs[0] * 100)))
    # ax.set_ylabel('PCA2 {}: ({}%)'.format(cols[pcs[1]], int(contribs[1] * 100)))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    # ax.set_xlim(-.2, +.65)
    # ax.set_ylim(-xylim, +xylim)
    # plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    # plt.legend()

    outpath = pjoin(outdir, 'pca.pdf')
    plt.savefig(outpath)

##########################################################
def plot_histograms(outdir):
    """Plot histograms"""
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

        W = 640; H = 480

        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        # ax.bar(range(1, n+1), np.array(counter) / np.sum(counter))
        ax.bar(range(1, n+1), np.array(counter))
        ax.set_xlabel('Feature id')
        ax.set_ylim(0, 9)
        plt.tight_layout()

        outpath = pjoin(outdir, 'meta_hist{}.pdf'.format(clid))
        plt.savefig(outpath)

##########################################################
def main(csvpath, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    df = pd.read_csv(csvpath)
    plot_pca(df, outdir)
    plot_histograms(outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csvpath', required=True, help='Path to the csv file')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.csvpath, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
