import pickle
import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform

from nilearn.maskers import MultiNiftiMapsMasker
from nilearn.datasets import fetch_atlas_difumo

from brainspace.gradient import GradientMaps


# Needs to be this many images to be considered
n_components = 5
connectivity_threshold = 5
connectivity_method = "correlation"

# Load one fmri sequence to get the labels
fmri_sequence = pd.read_csv(snakemake.input.sample_fmri_sequence)

# Make a conversion
index_to_image_name = {}
for i, row in fmri_sequence.iterrows():
    index_to_image_name[row["image_index"]] = row["image_path"].split("/")[-1]

container = {"learning": [], "subject": [], "image_index": []}
for subject_idx, subject in enumerate(snakemake.params.subjects):
    # Load the response order
    df_subject = pd.read_excel(
        snakemake.input.subject_responses[subject_idx],
        header=None, names=["image_name", "image_recognized"]
    )

    for image_index in range(33):

        # Get the image name
        img_name_this = index_to_image_name[image_index]

        # Match with the first row with this image name
        first_row = df_subject[df_subject["image_name"] == img_name_this]

        # Some names are missing a 1
        if len(first_row) == 0:
            img_name_this = img_name_this.replace(".bmp", "1.bmp")
            first_row = df_subject[df_subject["image_name"] == img_name_this]

        recs = first_row["image_recognized"].values
        rec_pre = recs[:len(recs) // 2].mean() >= 4/6
        unrec_pre = recs[:len(recs) // 2].mean() <= 3/6

        rec_post = recs[len(recs) // 2:].mean() >= 4/6
        unrec_post = recs[len(recs) // 2:].mean() <= 3/6

        container["subject"].append(subject)
        container["image_index"].append(image_index)

        if unrec_pre and unrec_post:
            container["learning"].append("Learned")
        elif unrec_pre and rec_post:
            container["learning"].append("Unlearned")
        else:
            if rec_pre:
                container["learning"].append("Recognized without learning")
            else:
                container["learning"].append(-1)

df = pd.DataFrame(container)

# Concat the data and mask
atlas_map = fetch_atlas_difumo(1024)['maps']
masker = MultiNiftiMapsMasker(atlas_map, smoothing_fwhm=2.0, standardize='zscore_sample').fit(snakemake.input.error)
error_masked = masker.transform(snakemake.input.error)
error_masked = np.concatenate(error_masked, axis=0)

# Fit the decomposition
n_datas = 0
all_data = {
    "Learned": 0,
    "Unlearned": 0,
    "Recognized without learning": 0
}
counter = {
    "Learned": 0,
    "Unlearned": 0,
    "Recognized without learning": 0
}
for (subject, learning), small_df in df.groupby(["subject", "learning"]):
    if learning == -1:
        continue

    data_this = error_masked[small_df.index]

    # If there is not enough data, skip
    if data_this.shape[0] < connectivity_threshold:
        continue

    # Find pairwise
    pairwise = 1-squareform(pdist(data_this.T, metric=connectivity_method))
    pairwise = np.nan_to_num(pairwise, nan=0.0)

    # Append
    all_data[learning] += pairwise
    counter[learning] += 1

# Average the data
all_data = {k: v/counter[k] for k, v in all_data.items()}

# Concat, decompose
grad = GradientMaps(
    n_components=n_components,
    kernel="normalized_angle",
    alignment="procrustes"
)
grad.fit([all_data[k] for k in ["Learned", "Unlearned", "Recognized without learning"]])

for idx in range(3):
    masker.inverse_transform(grad.aligned_[idx].T).to_filename(snakemake.output.grad_map[idx])

# Save the explained variance as well
for i, ev in enumerate(grad.lambdas_):
    with open(snakemake.output.explained_variance[i], "w") as f:
        # ev is a np array
        f.write(", ".join(ev.astype(str)))

# Store the pca and maskers
with open(snakemake.output.masker, "wb") as f:
    pickle.dump(masker, f)

with open(snakemake.output.grad, "wb") as f:
    pickle.dump(grad, f)