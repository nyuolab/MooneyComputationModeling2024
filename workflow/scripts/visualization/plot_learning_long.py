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


import wandb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


config = snakemake.config
model_records = snakemake.input.model_records

# Initialize wandb
wandb_run_name = config["central_logging_run_name"]
wandb.init(project=config["project_name"], entity="Aceticia", tags=config["tags"], name=wandb_run_name, resume="allow")

# Load each file
model_records = [pd.read_csv(f) for f in model_records]

# Concatenate all the files
data = pd.concat(model_records).reset_index(drop=True)

# Make figure
data["Accuracy"] = data["Ground truth class"] == data["Predicted class"]
ax = sns.lineplot(data=data, x="Time index", y="Accuracy", hue="Image phase", hue_order=["pre", "gray", "post", "repetition"], ci=95)

# Log the figure
plt.title("Long sequence learning")
wandb.log({"Long sequence learning": wandb.Image(plt)})

# Also save it
plt.savefig(snakemake.output.plot)
wandb.finish()