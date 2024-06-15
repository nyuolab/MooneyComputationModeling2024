import numpy as np
import pandas as pd
from scipy.io import loadmat

# Load the 
pre = loadmat(snakemake.input.pre_subject_record)["correct_pre"]
gray = loadmat(snakemake.input.gray_subject_record)["correct_g"]
post = loadmat(snakemake.input.post_subject_record)["correct_post"]

# Get a container
container = {
    "Source": [],
    "Detailed source": [],
    "Subject": [],
    "Image index": [],
    "Phase": [],
    "Correct": [],
    "Predicted": [],
}
for image in range(pre.shape[0]):
    for subject in range(pre.shape[1]):
        for phase, data in zip(["pre", "gray", "post"], [pre, gray, post]):
            correct = data[image][subject].item()
            container["Source"].append("Human")
            container["Detailed source"].append(f"Human-{subject}")
            container["Subject"].append(subject)
            container["Image index"].append(image)
            container["Phase"].append(phase)
            container["Correct"].append(correct)
            container["Predicted"].append("???")

# Load model data
for record_idx in range(len(snakemake.input.model_record)):
    loaded = pd.read_csv(snakemake.input.model_record[record_idx])
    subject, backbone, seed = snakemake.params.indicator[record_idx].split(";")
    for _, row in loaded.iterrows():
        container["Source"].append("Model")
        container["Detailed source"].append(f"Model-{seed}-{subject}")
        container["Subject"].append(subject)
        container["Image index"].append(row["Image index"])
        container["Phase"].append(row["Image phase"])
        container["Correct"].append("???")
        container["Predicted"].append(row["Predicted class"])

# Get the groundtruth class for each image
sample_df = pd.read_csv(snakemake.input.model_record[0])
sample_df = sample_df[["Image index", "Ground truth class"]].drop_duplicates()

# Make dataframe
df = pd.DataFrame(container)
df = df.merge(sample_df, on="Image index", how="left")

# Save
df.to_csv(snakemake.output[0], index=False)
