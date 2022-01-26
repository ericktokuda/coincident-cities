#!/usr/bin/env python3
"""one-line docstring
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

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')
    adjdir = '/tmp/cities5/'

    labels = [chr(i) for i in range(97, 97 + 21)]
    for f in sorted(os.listdir(adjdir)):
        if not f.endswith('.txt'): continue
        if 'README' in  f: continue
        info(f)
        adj = np.loadtxt(pjoin(adjdir, f))
        import seaborn as sns
        fig, ax = plt.subplots()
        mask = np.triu(np.ones_like(adj, dtype=bool))
        sns.heatmap(adj, mask=mask, ax=ax, vmin=-1, vmax=+1, xticklabels=labels, yticklabels=labels)
        plt.tight_layout()
        plt.savefig(pjoin(outdir, f.replace('.txt', '.pdf')))
        plt.close()

    info('For Aiur!')


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
