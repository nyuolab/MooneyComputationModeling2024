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


# include: "visualizations/attention_map.smk"
include: "visualizations/learning.smk"
include: "visualizations/recognition.smk"
include: "visualizations/attention_shift.smk"
include: "visualizations/repetition_effect.smk"


"""
For figure 1 we want the 
"""

rule fig_5b:
    input:
        expand(
            "results/long_native_trials/{synth_seed}_{backbone}_seed{seed}.csv",
            synth_seed=range(config["n_native_trials"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),


rule fig_5d:
    input:
        expand(
            "results/behavior_predictions/diff:{pred_source}:subject{sub}:{backbone}_seed{seed}_recognize_{phase}_{feature_type}.csv",
            pred_source="",
            sub=range(config["n_behavior_subjects"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
            phase=["pre", "post", "gray"],
            feature_type=config["feature_types"]+config["layer_feature_types"],
        )


rule fig_6a:
    input:
        # Maps of p-values
        expand(
            "results/encoding_significance/{phase}/pmap.nii",
            phase=["counterfactual-post"]
        ),
        expand(
            "results/encoding_significance/{phase}/thresholded_score_map.nii",
            phase=["counterfactual-post"]
        ),

        # ROI level summary
        "results/encoding_roi_comparison/comparisons.csv",
        expand(
            "results/roi_based_encoding_similarity/{phase}/performance_detailed.csv",
            phase=["counterfactual-post"]
        ),


rule fig_6c:
    input:
        expand(
            "results/prediction_error/counterfactual:post-post/subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        )


# Same as fig 6c
rule fig_6d:
    input:
        expand(
            "results/prediction_error/counterfactual:post-post/subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        )