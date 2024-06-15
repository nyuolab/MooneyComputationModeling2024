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


ruleorder: scramble_encoding > counterfactual_encoding > repetition_encoding > encoding

phase_to_control = {
    "post": "counterfactual",
    "gray": "scramble",
    "pre": "scramble"
}

rule get_encoding_utils:
    input:
        betas="results/fmris_standard/sub{subject}.nii.gz",
    output:
        mask="results/encoding_utils/{subject}/mask.nii",
        masked_data="results/encoding_utils/{subject}/masked_betas.npy",
        masker="results/encoding_utils/{subject}/masker.pkl",
    script:
        "../scripts/encoding/find_encoding_utils.py"


rule noise_ceiling:
    input:
        maps=expand("results/fmris_standard/sub{subject}.nii.gz", subject=config["fmri_subjects"]),
    output:
        lower="results/nc_lower_{phase}.nii.gz",
        upper="results/nc_upper_{phase}.nii.gz",
    script:
        "../scripts/encoding/noise_ceiling_map.py"


rule encoding:
    input:
        feat="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}_latent_vis.npy",
        sequence="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        betas="results/encoding_utils/{subject}/masked_betas.npy",
        mask="results/encoding_utils/{subject}/mask.nii",
        masker="results/encoding_utils/{subject}/masker.pkl",
    output:
        score="results/encoding/{model_phase}-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
        pred="results/encoding_predictions/{model_phase}-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
    script:
        "../scripts/encoding/encode_change.py"


rule repetition_encoding:
    input:
        feat="results/repetition_fmri_trials/human_subject{subject}:{backbone}_seed{seed}_latent_vis.npy",
        sequence="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        betas="results/encoding_utils/{subject}/masked_betas.npy",
        mask="results/encoding_utils/{subject}/mask.nii",
        masker="results/encoding_utils/{subject}/masker.pkl",
    output:
        score="results/encoding/repetition-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
        pred="results/encoding_predictions/repetition-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
    script:
        "../scripts/encoding/encode_change.py"


rule scramble_encoding:
    input:
        feat="results/scramble_fmri_trials/human_subject{subject}:{backbone}_seed{seed}_latent_vis.npy",
        sequence="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        betas="results/encoding_utils/{subject}/masked_betas.npy",
        mask="results/encoding_utils/{subject}/mask.nii",
        masker="results/encoding_utils/{subject}/masker.pkl",
    output:
        score="results/encoding/scramble-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
        pred="results/encoding_predictions/scramble-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
    script:
        "../scripts/encoding/encode_change.py"


rule counterfactual_encoding:
    input:
        feat="results/counterfactual_fmri_trials/human_subject{subject}:{backbone}_seed{seed}_latent_vis.npy",
        sequence="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        betas="results/encoding_utils/{subject}/masked_betas.npy",
        mask="results/encoding_utils/{subject}/mask.nii",
        masker="results/encoding_utils/{subject}/masker.pkl",
    output:
        score="results/encoding/counterfactual-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
        pred="results/encoding_predictions/counterfactual-{brain_phase}/human_subject{subject}:{backbone}_seed{seed}.nii.gz",
    script:
        "../scripts/encoding/encode_change.py"


rule avg_predictions:
    input:
        maps=expand("results/encoding_predictions/{{model_phase}}-{{brain_phase}}/human_subject{{subject}}:{backbone}_seed{seed}.nii.gz",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        avg_map="results/prediction_average/{model_phase}-{brain_phase}/human_subject{subject}.nii.gz",
    run:
        import numpy as np
        from nilearn.image import load_img, smooth_img, new_img_like
        template = load_img(input.maps[0])
        data = np.stack([load_img(m).get_fdata() for m in input.maps], axis=-1)
        mean = np.mean(data, axis=-1)
        new_img_like(template, mean, affine=template.affine).to_filename(output.avg_map)


rule avg_1st_level_enc:
    input:
        maps=expand("results/encoding/{{model_phase}}-{{brain_phase}}/human_subject{{subject}}:{backbone}_seed{seed}.nii.gz",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        avg_map="results/encoding_average_standard/{model_phase}-{brain_phase}/human_subject{subject}.nii.gz",
    run:
        from nilearn.image import mean_img
        mean_img(input.maps).to_filename(output.avg_map)


rule avg_2nd_level_enc:
    input:
        maps=expand("results/encoding_average_standard/{{model_phase}}-{{brain_phase}}/human_subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
    output:
        avg_map="results/encoding_significance/{model_phase}-{brain_phase}/avg_sim.nii",
    run:
        import numpy as np
        from nilearn.image import load_img, smooth_img, new_img_like
        template = load_img(input.maps[0])
        data = np.stack([smooth_img(m, 5.0).get_fdata() for m in input.maps], axis=-1)
        mean = np.tanh(np.mean(data, axis=-1))
        new_img_like(template, mean, affine=template.affine).to_filename(output.avg_map)


rule find_significance_map_enc:
    input:
        pos_maps=expand("results/encoding_average_standard/{{brain_phase}}-{{brain_phase}}/human_subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
        neg_maps=expand("results/encoding_average_standard/{{control_phase}}-{{brain_phase}}/human_subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
        avg_map="results/encoding_significance/{brain_phase}-{brain_phase}/avg_sim.nii",
        avg_map_control="results/encoding_significance/{control_phase}-{brain_phase}/avg_sim.nii",
    output:
        thresholded_score="results/encoding_significance/{control_phase}-{brain_phase}/thresholded_score_map.nii",
        pmap="results/encoding_significance/{control_phase}-{brain_phase}/pmap.nii",
        thresholded_pmap="results/encoding_significance/{control_phase}-{brain_phase}/thresholded_pmap.nii",
        tmap="results/encoding_significance/{control_phase}-{brain_phase}/tmap.nii",
        cluster_record="results/encoding_significance/{control_phase}-{brain_phase}/cluster_record.csv",
    script:
        "../scripts/encoding/compare_with_control.py"


rule find_where_below_nc:
    input:
        maps=expand("results/encoding_average_standard/{{model_phase}}-{{brain_phase}}/human_subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
        nc_map="results/nc_lower_{brain_phase}.nii.gz",
    output:
        pmap="results/below_nc_significance/{model_phase}-{brain_phase}/pmap.nii",
        tmap="results/below_nc_significance/{model_phase}-{brain_phase}/tmap.nii",
        cluster_record="results/below_nc_significance/{model_phase}-{brain_phase}/cluster_record.csv",
    params:
        lower=True
    script:
        "../scripts/encoding/compare_against_nc.py"


rule find_where_above_nc:
    input:
        maps=expand("results/encoding_average_standard/{{model_phase}}-{{brain_phase}}/human_subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
        nc_map="results/nc_lower_{brain_phase}.nii.gz",
    output:
        pmap="results/above_nc_significance/{model_phase}-{brain_phase}/pmap.nii",
        tmap="results/above_nc_significance/{model_phase}-{brain_phase}/tmap.nii",
        cluster_record="results/above_nc_significance/{model_phase}-{brain_phase}/cluster_record.csv",
    params:
        lower=False
    script:
        "../scripts/encoding/compare_against_nc.py"


rule roi_based_encoding_similarity:
    input:
        score1="results/encoding_significance/{brain_phase}-{brain_phase}/avg_sim.nii",
        score2="results/encoding_significance/{control_phase}-{brain_phase}/avg_sim.nii",
        pmap="results/encoding_significance/{control_phase}-{brain_phase}/pmap.nii",
        roi_masks=expand(
            "data/beta_values/standard_brain_rois/{roi}.nii.gz",
            roi=config['standard_rois']
        )
    params:
        roi_names=config['standard_rois']
    output:
        score="results/roi_based_encoding_similarity/{control_phase}-{brain_phase}/performance_detailed.csv",
    script:
        "../scripts/encoding/roi_evaluation_detailed.py"


rule get_connectome_surface:
    input:
        connectome_L="data/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
        connectome_R="data/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
        thresholded_map="results/encoding_significance/{control_phase}-{brain_phase}/thresholded_score_map.nii"
    output:
        surface_data_L="results/thresholded_map/{control_phase}-{brain_phase}/surface_data_L.shape.gii",
        surface_data_R="results/thresholded_map/{control_phase}-{brain_phase}/surface_data_R.shape.gii",
    shell:
	    """
	    wb_command -volume-to-surface-mapping {input.thresholded_map} {input.connectome_L} {output.surface_data_L} -trilinear
	    wb_command -volume-to-surface-mapping {input.thresholded_map} {input.connectome_R} {output.surface_data_R} -trilinear
	    """