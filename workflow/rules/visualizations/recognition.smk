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


rule visualize_diff_rec_pred:
    input:
        diff_model_records=expand(
            "results/behavior_predictions/diff:human_subject{sub}:subject{sub}:{backbone}_seed{seed}_recognize_{{phase}}_{{feature_type}}.csv",
            sub=range(config["n_behavior_subjects"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    params:
        behavior_name="recognition",
    group: "plotting"
    output:
        plot="results/visualization/recognition_diff_prediction/+all_models_{phase}_{feature_type}.jpg",
    script:
        "../../scripts/visualization/plot_diff_predictions.py"