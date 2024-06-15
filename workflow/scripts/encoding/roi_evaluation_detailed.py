import numpy as np
import pandas as pd

from collections import defaultdict

from nilearn.maskers import NiftiMasker
from nilearn.image import load_img, binarize_img

roi_threshold = 0.5
p_threshold = 0.05
neg_log_p_threshold = -np.log10(p_threshold)

target_affine = load_img(snakemake.input.score1).affine

container = defaultdict(list)
for roi_mask, roi_name in zip(snakemake.input.roi_masks, snakemake.params.roi_names):
    for score_name, score_type in [
        ("Control score", snakemake.input.score1),
        ("Encoding score", snakemake.input.score2),
    ]:
        # Get the roi mask
        masker = NiftiMasker(
            mask_img=binarize_img(roi_mask, threshold=roi_threshold),
            target_affine=target_affine,
            standardize=False
        ).fit()

        # Get the roi mask
        score_this = masker.transform(score_type)
        p_val = masker.transform(snakemake.input.pmap)

        # If too small, skip
        if score_this.size < 50:
            continue

        # Find significant voxels
        significant = p_val > neg_log_p_threshold
        all_voxels = np.ones_like(significant)

        # Percentage of significant voxels
        significant_percentage = np.mean(significant)

        # Get these
        container["roi_name"].append(roi_name)
        container["roi_size"].append(significant.size)
        container["percentage_significant"].append(significant_percentage)
        container["score_type"].append(score_name)

        for significance_type, criterion in [("all", significant), ("significant", all_voxels)]:
            for aggregation_type, agg_func in [("mean", np.mean), ("median", np.median), ("variance", np.var), ("max", np.max)]:
                if np.any(criterion):
                    container[f"{significance_type}_{aggregation_type}_score"].append(agg_func(score_this[criterion]))
                else:
                    container[f"{significance_type}_{aggregation_type}_score"].append(np.nan)

# Create a dataframe
pd.DataFrame(container).to_csv(snakemake.output[0], index=False)