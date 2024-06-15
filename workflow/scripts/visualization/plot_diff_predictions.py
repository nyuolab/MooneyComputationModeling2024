import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


config = snakemake.config
behavior_name = snakemake.params.behavior_name
diffs = snakemake.input.diff_model_records
phase = snakemake.wildcards.phase

# Always plot all models
feature_type = snakemake.wildcards.feature_type

# Load each file
model_records = [pd.read_csv(f) for f in diffs]

# Concatenate all the files
data = pd.concat(model_records).reset_index()

# Only keep relevant columns
data = data[["subject", "perf_type", f"{feature_type}_perf"]]
data = data.rename(columns={f"{feature_type}_perf": "Accuracy", "subject": "Subject", "perf_type": "Prediction type"})

# Plot the accuracy
ax = sns.barplot(data=data, x="Subject", y="Accuracy", hue="Prediction type", hue_order=["CV", "Chance"])

# Annotate difference
pairs = [((subject, "CV"), (subject, "Chance")) for subject in data["Subject"].unique()]
annotator = Annotator(ax, pairs, data=data, x="Subject", y="Accuracy", hue="Prediction type", hue_order=["CV", "Chance"])
annotator.configure(test='Mann-Whitney-gt', text_format='star', comparisons_correction="BH")
annotator.apply_and_annotate()

title = f"{phase} {feature_type} feature for predicting {behavior_name}"
plt.title(title)
plt.savefig(snakemake.output.plot)
