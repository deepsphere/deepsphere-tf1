"""
    Script from Carlos Esteves, used for the SHREC17 competion
"""

import os
import subprocess
from joblib import Parallel, delayed
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_curve, precision_score

from spherical_cnn import models, util


def make_shrec17_output_thresh(descriptors, scores, fnames, outdir,
                               distance='cosine', dists=None, thresh=None):
    if dists is None:
        dists = squareform(pdist(descriptors, distance))
    fnames = [os.path.splitext(f)[0] for f in fnames]
    os.makedirs(outdir, exist_ok=True)

    if not isinstance(thresh, dict):
        thresh = {i: thresh for i in range(scores.shape[1])}

    predclass = scores.argmax(axis=1)

    lens = Parallel(n_jobs=-1)(delayed(make_shrec17_output_thresh_loop)
                               (d, f, s, c, thresh, fnames, predclass, outdir)
                               for d, f, s, c in zip(dists, fnames, scores, predclass))

    print('avg # of elements returned {:2f} {:2f}'.format(np.mean(lens), np.std(lens)))


def make_shrec17_output_thresh_loop(d, f, s, c, thresh, fnames, predclass, outdir, max_retrieved=1000):
        t = thresh[c]

        fd = [(ff, dd)
              for dd, ff, cc in zip(d, fnames, predclass)
              # chose whether to include same class or not
              if (dd < t) or (cc == c)]
              # if (dd < t)]
        fi = [ff[0] for ff in fd]
        di = [ff[1] for ff in fd]

        ranking = []
        for i in np.argsort(di):
            if fi[i] not in ranking:
                ranking.append(fi[i])
        ranking = ranking[:max_retrieved]

        with open(os.path.join(outdir, f), 'w') as fout:
            [print(r, file=fout) for r in ranking]

        return len(ranking)


def make_shrec17_output(descriptors, scores, fnames, outdir,
                        distance='cosine', dists=None,
                        max_retrieved=1000):
    if dists is None:
        dists = squareform(pdist(descriptors, distance))
    fnames = [os.path.splitext(f)[0] for f in fnames]
    os.makedirs(outdir, exist_ok=True)

    predclass = scores.argmax(axis=1)
    for d, f, s in zip(dists, fnames, scores):
        # return elements from top nc classes
        nc = 1
        cs = np.argsort(s)[::-1][:nc]

        # list elements of the selected classes and its distances
        fi, di = [], []
        for c in cs:
            fi += [ff for ff, cc in zip(fnames, predclass) if cc == c]
            di += [dd for dd, cc in zip(d, predclass) if cc == c]

        # also include elements with distance less than the median
        median = np.median(di)
        fi += [ff for ff, dd in zip(fnames, d) if dd < median]
        di += [dd for dd in d if dd < median]

        # return unique entries !!!
        ranking = []
        for idx in np.argsort(di):
            if fi[idx] not in ranking:
                ranking.append(fi[idx])
        ranking = ranking[:max_retrieved]

        with open(os.path.join(outdir, f), 'w') as fout:
            [print(r, file=fout) for r in ranking]


def eval_shrec17_output(outdir):
    basedir = Path(os.path.realpath(__file__)).parent / '..'
    evaldir = basedir / 'external/shrec17_evaluator'
    assert basedir.is_dir()
    assert evaldir.is_dir()
    assert os.path.isdir(outdir)
    evaldir = str(evaldir)
    # import ipdb; ipdb.set_trace()
    if outdir[-1] != '/':
        outdir += '/'
    # outdir_arg = os.path.join('../../', outdir)
    p = subprocess.Popen(['node', 'evaluate.js', outdir],
                         cwd=evaldir)
    p.wait()

    import pandas as pd
    data = pd.read_csv('{}/{}.summary.csv'
                       .format(evaldir, outdir.split('/')[-2]))

    return data


def save_descriptors_dists(modeldir, dset_fname, ckpt='best.ckpt'):
    """ Save descriptors and pairwise distances. """
    layers = ['descriptor', 'out']
    # can only use labels on val or train!
    layers += ['label']
    out = models.get_tfrecord_activations(modeldir, dset_fname, layers, ckptfile=ckpt,
                                          args_in={'test_bsize': 32, 'train_bsize': 32})

    out['d_cosine'] = squareform(pdist(out['descriptor'], 'cosine'))
    fname = Path(dset_fname).parts[-1].split('.')[0]
    np.savez(os.path.join(modeldir, '{}_descriptors_scores.npz'.format(fname)), **out)

    return out


def search_thresholds(dists_or_file):
    """ Search thresholds per class that maximizes F-score. """

    if isinstance(dists_or_file, str):
        out = np.load()
    else:
        out = dists_or_file

    dists = out['d_cosine']
    labels = out['label']

    thresh = {i: [] for i in range(max(labels)+1)}
    dists /= dists.max()
    assert dists.min() >= 0
    assert dists.max() <= 1

    list_thresh = Parallel(n_jobs=-1)(delayed(search_thresholds_loop)(d, l, labels) for d, l in zip(dists, labels))

    for l, t in zip(labels, list_thresh):
        thresh[l].append(t)

    # mean thresh per class
    # these are 1-d, need to be more than that to be classified
    # d must be smaller than 1-this value
    thresh_mean = {i: 1-np.mean(t) for i, t in sorted(thresh.items())}

    return thresh_mean


def search_thresholds_loop(d, l, labels):
    p, r, t = precision_recall_curve(labels == l, 1-d)
    f = 2 * (p * r) / (p + r)

    return t[np.argmax(f)]


def run_all_shrec17(modeldir, datadir, ckpt):
    """ Run all steps for retrieval.

    Compute descriptors, distances, thresholds, shrec17 output and evaluate. """

    fnames = os.path.join(datadir, '{}0.tfrecord')
    out = save_descriptors_dists(modeldir, fnames.format('train'), ckpt)
    thresh = search_thresholds(out)

    # and test set to evaluate models
    out = save_descriptors_dists(modeldir, fnames.format('test'), ckpt)
    fnames = util.tfrecord_fnames(fnames.format('test'))

    # trailing slash required
    descr, scores, dists = out['descriptor'], out['out'], out['d_cosine']
    outdir = '/tmp/ranking_{}_norepeat/test_perturbed'.format(os.path.split(modeldir)[-1])
    assert not os.path.isdir(outdir)
    make_shrec17_output_thresh(descr, scores, fnames, outdir,
                               distance='cosine', dists=dists, thresh=thresh)

    res = eval_shrec17_output(os.path.split(outdir)[0])
    print(modeldir, datadir, ckpt)
    print(res.head(1))
    print(res.tail(1))    

    return res
