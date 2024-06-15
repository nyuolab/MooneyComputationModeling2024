import numpy as np
from scipy.io import loadmat, savemat

pre, post = snakemake.input.pre, snakemake.input.post

# Are we dealing with RT or recognition?
mode = snakemake.wildcards.misc

# Load pre and post
pre = loadmat(pre, struct_as_record=True)[f'{mode}_pre']
post = loadmat(post, struct_as_record=True)[f'{mode}_post']

# Iterate over subjects
diff_result = np.zeros((pre.shape[0], pre.shape[1]))
for subject in range(pre.shape[1]):
    for image in range(pre.shape[0]):
        try:
            tmp_pre = pre[image][subject].item()
            tmp_post = post[image][subject].item()

            # Take difference
            if mode == "rt":
                diff_result[image, subject] = tmp_post - tmp_pre
            else:
                diff_result[image, subject] = np.nan if tmp_pre else tmp_post
        except ValueError:
            diff_result[image, subject] = np.nan

# Save to mat
savemat(snakemake.output.diff, {f'{mode}_diff': diff_result})