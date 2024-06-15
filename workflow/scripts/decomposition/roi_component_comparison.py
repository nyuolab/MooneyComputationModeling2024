# This file is part of Mooney computational modeling project.
#
# Mooney computational modeling project is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mooney computational modeling project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mooney computational modeling project. If not, see <https://www.gnu.org/licenses/>.


import numpy as np

from nilearn.image import smooth_img, math_img, resample_to_img
from nilearn.masking import compute_brain_mask
from nilearn.regions import RegionExtractor

from scipy.stats import pearsonr, rankdata

from neuromaps.images import load_data
from sklearn.utils.validation import check_random_state

eps = 1e-3


# We rewrite the function here, since we are only looking for positive side of correlation
def compare_images(src, trg, metric, ignore_zero=True, nulls=None,
                   nan_policy='omit', return_nulls=False):
    if return_nulls and nulls is None:
        raise ValueError('`return_nulls` cannot be True when `nulls` is None.')

    srcdata, trgdata = load_data(src), load_data(trg)

    # drop NaNs (if nan_policy==`omit`) and zeros (if ignore_zero=True)
    zeromask = np.zeros(len(srcdata), dtype=bool)
    if ignore_zero:
        # Only exlucde if both are zeros
        zeromask = np.logical_and(np.isclose(srcdata, 0),
                                 np.isclose(trgdata, 0))
    nanmask = np.logical_or(np.isnan(srcdata), np.isnan(trgdata))
    if nan_policy == 'raise':
        if np.any(nanmask):
            raise ValueError('Inputs contain nan')
    elif nan_policy == 'omit':
        mask = np.logical_and(np.logical_not(zeromask),
                              np.logical_not(nanmask))
    elif nan_policy == 'propagate':
        mask = np.logical_not(zeromask)
    srcdata, trgdata = srcdata[mask], trgdata[mask]

    if nulls is not None:
        n_perm = nulls.shape[-1]
        nulls = nulls[mask]
        return permtest_metric(srcdata, trgdata, metric, n_perm=n_perm,
                               nulls=nulls, nan_policy=nan_policy,
                               return_nulls=return_nulls)

    return metric(srcdata, trgdata)


def permtest_metric(a, b, metric, n_perm=1000, seed=0, nulls=None,
                    nan_policy='propagate', return_nulls=False):

    def nan_wrap(a, b, nan_policy='propagate'):
        nanmask = np.logical_or(np.isnan(a), np.isnan(b))
        if nan_policy == 'raise':
            if np.any(nanmask):
                raise ValueError('Input contains nan')
        elif nan_policy == 'omit':
            a, b = a[~nanmask], b[~nanmask]
        return metric(a, b)

    rs = check_random_state(seed)

    if len(a) != len(b):
        raise ValueError('Provided arrays do not have same length')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if nulls is not None:
        n_perm = nulls.shape[-1]

    # divide by one forces coercion to float if ndim = 0
    true_sim = nan_wrap(a, b, nan_policy=nan_policy) / 1

    permutations = np.ones(true_sim.shape)
    nulldist = np.zeros(((n_perm, ) + true_sim.shape))
    for perm in range(n_perm):
        # permute `a` and determine whether correlations exceed original
        ap = a[rs.permutation(len(a))] if nulls is None else nulls[:, perm]
        nullcomp = nan_wrap(ap, b, nan_policy=nan_policy)
        permutations += nullcomp >= true_sim
        nulldist[perm] = nullcomp

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_sim

    if return_nulls:
        return true_sim, pvals, nulldist

    return true_sim, pvals

# Load roi map. Smooth it a bit to get better transition
roi_map = smooth_img(snakemake.input.roi, fwhm=5.0)
roi_map = resample_to_img(roi_map, snakemake.input.component)

# Add a small value to all the voxels inside brain
brain_mask = compute_brain_mask(roi_map, threshold=0.1, connected=False)
roi_map = math_img(f"img + brain * {eps}", img=roi_map, brain=brain_mask)

# Define a correlation comparison metric that uses the magnitude of inputs
def tie_corrected_spearmanr(x, y):
    x = rankdata(x)
    y = rankdata(y)
    x = x - x.mean()
    y = y - y.mean()
    n = len(x)
    return np.dot(x, y) / (n ** 3 - n) * 12


def soft_dice(x, y):
    # Remove eps
    y = y - eps
    return 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))


def percent(c1, c2, p=2):
    total_energy = np.linalg.norm(c1, ord=p)
    return np.linalg.norm(c1[c2 > eps], ord=p) / total_energy


def density(c1, c2):
    # Threshold roi
    roi_yes = c2 > eps

    # Calculate total, and over the roi size
    total = np.sum(c1[roi_yes])
    return total / np.sum(c1)


def coverage(c1, c2):
    # Threshold roi
    roi_yes = c2 > eps

    # Threshold component
    component_threshold = np.percentile(c1, 90)
    component_yes = c1[roi_yes] > component_threshold

    return np.mean(component_yes)


metric = {
    "pearsonr": lambda x, y: pearsonr(x, y)[0],
    "spearmanr": tie_corrected_spearmanr,
    "dice": soft_dice,
    "percent": percent,
    "density": density,
    "coverage": coverage
}[snakemake.params.method]


# Set left and right values
all_val = compare_images(
    src=smooth_img(snakemake.input.component, 5.0), trg=roi_map,
    nulls=np.load(snakemake.input.null), metric=metric, nan_policy='omit'
)
np.save(snakemake.output.comparison, all_val)