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


import pickle as pkl
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from nilearn.image import concat_imgs, math_img, new_img_like


rule concat_reorder_fmri:
    input:
        fmris=[
            "data/beta_values/sub{subject}/" + f"beta_{conditions:04}.nii"
            for conditions in range(1, config["n_conditions"] + 1)
        ],
        regressor_orders="data/beta_values/sub{subject}/regressor_names.mat",
        target_order="data/label_order_target.txt",
    group: "concat"
    output:
        temp("results/fmris/sub{subject}.nii"),
    run:
        this_order = loadmat(input.regressor_orders)["regressor_names"][0]
        this_names = [str(x).split("'")[1] for x in this_order if "SPM" not in str(x)]
        target_order = open(input.target_order, "r").read().split("\n")
        take_idxs = [this_names.index(target_name) for target_name in target_order]
        img = concat_imgs([input.fmris[take_idx] for take_idx in take_idxs])
        img.to_filename(str(output))


rule get_standard_beta:
    input:
        reference="data/beta_values/MNI152_T1_2mm_brain_mask.nii.gz",
        beta_map="data/beta_values/sub{subject}/beta_{condition}.nii",
        conversion_init="data/beta_values/sub{subject}/example_func2standard.mat",
    output:
        out_map="results/fmris_standard/sub{subject}_{condition}.nii.gz"
    run:
        applyxfm(
            src=input.beta_map,
            ref=input.reference,
            mat=input.conversion_init,
            interp="trilinear",
            out=output.out_map,
        )


rule concat_reorder_standard_fmri:
    input:
        fmris=[
            "results/fmris_standard/sub{subject}_" + f"{conditions:04}.nii.gz"
            for conditions in range(1, config["n_conditions"] + 1)
        ],
        regressor_orders="data/beta_values/sub{subject}/regressor_names.mat",
        target_order="data/label_order_target.txt",
    group: "concat"
    output:
        temp("results/fmris_standard/sub{subject}.nii.gz"),
    run:
        this_order = loadmat(input.regressor_orders)["regressor_names"][0]
        this_names = [str(x).split("'")[1] for x in this_order if "SPM" not in str(x)]
        target_order = open(input.target_order, "r").read().split("\n")
        take_idxs = [this_names.index(target_name) for target_name in target_order]
        img = concat_imgs([input.fmris[take_idx] for take_idx in take_idxs])
        img.to_filename(str(output))


rule read_human_fmri_sequence:
    input:
        classes="data/fmri_classes.txt",
        response_order="data/fmri_image_order/S{subject}/response_order.xlsx",
    output:
        image_order="results/fmri_sequences/raw_human_sequence_{subject}.txt",
    group: "human_fmri"
    run:
        import pandas as pd
        filenames = pd.read_excel(input.response_order, header=None)[0].values
        with open(output.image_order, "w") as f:
            f.write("\n".join(filenames))


rule parse_human_fmri_sequence:
    input:
        classes="data/fmri_classes.txt",
        image_filenames="data/fmri_filenames.txt",
        image_order="results/fmri_sequences/raw_human_sequence_{subject}.txt",
    group: "human_fmri"
    output:
        "results/fmri_sequences/human_subject{subject}.csv",
    params:
        mode="fmri",
    script:
        "../scripts/parse_human_sequence.py"


rule fmri_trial_real:
    input:
        checkpoint="results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
        behavior_sequence="results/fmri_sequences/human_subject{subject}.csv",
    resources:
        nvidia_gpu=1,
    retries: 100
    output:
        behaviors="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        representations=temp(expand(
            "results/fmri_trials/human_subject{{subject}}:{{backbone}}_seed{{seed}}_{feature}.npy",
            feature=config["feature_types"]+config["layer_feature_types"]+config["static_feature_types"]+config["accumulate_feature_types"]
        )),
    script:
        "../scripts/model_behaviors/behavior_trial.py"


rule make_repetition_sequences:
    input:
        behavior_sequence="results/fmri_sequences/human_subject{subject}.csv",
    output:
        rep_beh_seq="results/repetition_fmri_sequences/human_subject{subject}.csv",
    run:
        import pandas as pd
        df = pd.read_csv(input.behavior_sequence)
        df["image_path"] = df["image_path"].str.replace("filter", "test")
        df.to_csv(output.rep_beh_seq, index=False)


rule fmri_trial_repetition:
    input:
        checkpoint="results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
        behavior_sequence="results/repetition_fmri_sequences/human_subject{subject}.csv",
    resources:
        nvidia_gpu=1,
    output:
        behaviors="results/repetition_fmri_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        representations=temp(expand(
            "results/repetition_fmri_trials/human_subject{{subject}}:{{backbone}}_seed{{seed}}_{feature}.npy",
            feature=config["feature_types"]+config["layer_feature_types"]+config["static_feature_types"]+config["accumulate_feature_types"]
        )),
    script:
        "../scripts/model_behaviors/behavior_trial.py"


rule counterfactual_model_feat:
    input:
        catch_images=expand("data/catch_images/{image}.bmp", image=range(31)),
        repetition_feat="results/repetition_fmri_trials/human_subject{subject}:{backbone}_seed{seed}_latent_vis.npy",
        checkpoint="results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
        behavior_sequence="results/fmri_sequences/human_subject{subject}.csv",
    resources:
        nvidia_gpu=1,
    output:
        state="results/counterfactual_fmri_trials/human_subject{subject}:{backbone}_seed{seed}_latent_vis.npy"
    script:
        "../scripts/model_behaviors/counterfactual_fmri_trial.py"