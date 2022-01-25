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
import pandas as pd
from myutils import info, create_readme
from myutils.transform import pca, get_pc_contribution

##########################################################
def main(csvpath, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    df = pd.read_csv(csvpath)

    cols = [ 'degmean', 'degstd', 'deg3', 'deg4', 'deg5', 'transmean', 'transstd', 'eangstd',
    'vposstd2', 'lacun21', 'acc05mean', 'acc05std']

    df2 = df[cols].copy(deep=True)
    data = df2.to_numpy()
    tr, evecs, evals = pca(data, normalize=True)
    pcs, contribs = get_pc_contribution(evecs)

    W = 640; H = 480

    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    # ax.set_title('PCA components (pc1 and pc2)')

    cities = [ c.capitalize() for c in list(df.city)]

    ax.scatter(tr[:, 0], tr[:, 1])
    # for i in range(tr.shape[0]):
        # ax.scatter(tr[i, 0], tr[i, 1], label=cities[i])

    for i in range(len(df2)):
        ax.annotate(df.iloc[i].city, (tr[i, 0], tr[i, 1]))

    xylim = np.max(np.abs(tr[:, 0:2])) * 1.1
    # ax.set_xlabel('PCA1 ({} ({}%)'.format(cols[pcs[0]], int(contribs[0] * 100)))
    # ax.set_ylabel('PCA2 {}: ({}%)'.format(cols[pcs[1]], int(contribs[1] * 100)))
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    # ax.set_xlim(-.2, +.65)
    # ax.set_ylim(-xylim, +xylim)
    # plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    # plt.legend()

    outpath = '/tmp/pca.png'
    plt.savefig(outpath)

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
