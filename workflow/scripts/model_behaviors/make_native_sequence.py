import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import pandas as pd
import numpy as np

from datasets import load_dataset

config = snakemake.config

# Number of grayscales to take
n_images = 210

# Take this number of images from the validation set
np.random.seed(int(snakemake.wildcards.synth_seed))

# Get the model's class names
val_set = load_dataset(config["dataset"], split=config["val_split_name"], cache_dir=config["cache_dir"])


gray_order = np.random.choice(len(val_set), size=n_images, replace=False).tolist()

# First decide the order of grayscale to use
assert n_images % 3 == 0

# Prepare for the output csv
image_phase = []
image_index = []
image_class = []

# Assign pre_post bucket and gray bucket
gray_bucket = []
pre_post_bucket = []
pre_post_phase_bucket = []
for sequence_idx, gray_start in enumerate(range(0, n_images, 3)):
    gray_involed_curr = gray_order[gray_start:gray_start+3]

    # Put pre in if we are not at the end
    if sequence_idx != n_images // 3:
        pre_post_bucket.extend(gray_involed_curr)
        pre_post_phase_bucket.extend(['pre'] * 3)

    # Put gray in
    gray_bucket.extend(gray_involed_curr)

    # Shuffle gray and pre_post bucket
    pre_post_perm = np.random.permutation(len(pre_post_bucket))
    pre_post_bucket = [pre_post_bucket[i] for i in pre_post_perm]
    pre_post_phase_bucket = [pre_post_phase_bucket[i] for i in pre_post_perm]
    gray_bucket = np.random.permutation(gray_bucket).tolist()

    # Write pre_post bucket
    image_phase.extend(pre_post_phase_bucket)
    image_index.extend(pre_post_bucket)
    image_class.extend([val_set[image_idx]["label"] for image_idx in pre_post_bucket])

    # Write gray bucket
    image_phase.extend(["gray"] * 3)
    image_index.extend(gray_bucket)
    image_class.extend([val_set[image_idx]["label"] for image_idx in gray_bucket])

    # Clean the buckets for next iteration, and write the next post bucket
    pre_post_bucket = gray_involed_curr
    pre_post_phase_bucket = ["post"] * 3
    gray_bucket = []

# Write post bucket one last time
image_phase.extend(pre_post_phase_bucket)
image_index.extend(pre_post_bucket)
image_class.extend([val_set[image_idx]["label"] for image_idx in pre_post_bucket])

# Record the sequence to use
pd.DataFrame({
    "Image phase": image_phase,
    "Image index": image_index,
    "Ground truth class": image_class,
}).to_csv(snakemake.output.sequence, index=False)