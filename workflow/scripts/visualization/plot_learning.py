"""
Plot whether learning happens.
Just plot pre, gray, post accuracies with all t together.
Use statannotations to annotate the diff from pre and post.

Make the same plot for each individual models, and then for all.
"""

import wandb
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt


config = snakemake.config
model_records = snakemake.input.model_records
wandb_run_name = config["central_logging_run_name"]

# Initialize wandb
wandb.init(project=config["project_name"], entity="Aceticia", tags=config["tags"], name=wandb_run_name, resume="allow")

# Load each file
model_records = [pd.read_csv(f) for f in model_records]

# Concatenate all the files
data = pd.concat(model_records).reset_index(drop=True)

# Add a column for the correct predictions
data["correct"] = data["Predicted class"] == data["Ground truth class"]

# Make figure
data = data.rename(columns={"correct": "Accuracy"})
ax = sns.barplot(data=data, x="Image phase", y="Accuracy", order=["pre", "post", "gray"])

# Annotate
pairs = [["pre", "post"], ["post", "gray"]]
annotator = Annotator(ax, pairs, data=data, x="Image phase", y="Accuracy", order=["pre", "post", "gray"])
annotator.configure(test='Mann-Whitney', text_format='star', comparisons_correction="BH")
annotator.apply_and_annotate()

# Log the figure

if hasattr(snakemake.wildcards, "extra"):
    extra = snakemake.wildcards.extra
else:
    extra = ""

plt.title(f"Learning in pre, gray, post phase {extra}")
wandb.log({f"PrePost {extra}": wandb.Image(plt)})

# Also save it
plt.savefig(snakemake.output.plot)

# Finally, log the average accuracy to wandb
wandb.log({"PrePostAvg": data.groupby("Image phase")["Accuracy"].mean().to_dict()})
wandb.finish()