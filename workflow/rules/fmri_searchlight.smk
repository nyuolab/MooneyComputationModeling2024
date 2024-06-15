import os
from itertools import product
import numpy as np
from nilearn.image import new_img_like, math_img
from fsl.wrappers import applyxfm


rule model_rdms:
    input:
        sequences="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        features="results/fmri_trials/human_subject{subject}:{backbone}_seed{seed}_{feat}.npy",
    group: "model_rdm"
    output:
        all_rdm="results/model_rdms/human_subject{subject}:{backbone}_seed{seed}_{feat}.npy",
    script:
        "../scripts/model_rdm_critical.py"


rule get_searchlight_rdms:
    input:
        betas="results/fmris/sub{subject}.nii",
        center="results/searchlight_utils/{subject}_searchlight_center.pkl",
        neighbs="results/searchlight_utils/{subject}_searchlight_neighbs.pkl",
    params:
        radius=config["searchlight_ball_radius"],
        threshold=config["searchlight_ball_threshold"],
    output:
        temp("results/searchlight_rdms/sub{subject}_{phase}.pkl"),
    script:
        "../scripts/searchlight/get_searchlight_rdms.py"


rule searchlight_sim:
    input:
        template="results/fmris/sub{subject}.nii",
        searchlight_rdms="results/searchlight_rdms/sub{subject}_{phase}.pkl",
        model_rdms="results/model_rdms/human_subject{subject}:{backbone}_seed{seed}_{feat}.npy"
    output:
        sim_map="results/searchlight_sim_func/{phase}_{subject}_{backbone}_{seed}_{feat}.nii.gz"
    script:
        "../scripts/searchlight/searchlight_sim_combined.py"


rule searchlight_t:
    input:
        maps=expand(
            "results/searchlight_sim_func/{{phase}}_{{subject}}_{backbone}_{seed}_{{feat}}.nii.gz",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        tmap="results/searchlight_t_func/{feat}-{phase}-{subject}.nii.gz",
    run:
        import numpy as np
        from nilearn.image import load_img
        template = load_img(input.maps[0])
        arr = np.stack([load_img(m).get_fdata() for m in input.maps], axis=-1)
        mean = np.mean(arr, axis=-1)
        std_dev = np.std(arr, axis=-1, ddof=1)
        n = arr.shape[-1]
        t_stat = mean / (std_dev / np.sqrt(n))
        new_img_like(template, t_stat, affine=template.affine).to_filename(output.tmap)


rule avg_similarity:
    input:
        maps=expand(
            "results/searchlight_sim_func/{{phase}}_{{subject}}_{backbone}_{seed}_{{feat}}.nii.gz",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        avg_map="results/searchlight_sim_func/avg_{feat}-{phase}-{subject}.nii.gz",
    run:
        import numpy as np
        from nilearn.image import load_img
        template = load_img(input.maps[0])
        data = np.stack([load_img(m).get_fdata() for m in input.maps], axis=-1)
        mean = np.mean(data, axis=-1)
        new_img_like(template, mean, affine=template.affine).to_filename(output.avg_map)


rule apply_transform:
    input:
        reference="data/beta_values/MNI152_T1_2mm_brain_mask.nii.gz",
        map="results/searchlight_{misc}_func/{misc2}-{subject}.nii.gz",
        conversion_init="data/beta_values/sub{subject}/example_func2standard.mat",
    output:
        out_map="results/searchlight_{misc}_standard/{misc2}-{subject}.nii.gz",
    run:
        applyxfm(
            src=input.map,
            ref=input.reference,
            mat=input.conversion_init,
            interp="trilinear",
            out=output.out_map,
        )


rule avg_2nd_level:
    input:
        maps=expand(
            "results/searchlight_sim_standard/avg_{{feat}}-{{phase}}-{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
    output:
        avg_map="results/searchlight_significance/{feat}-{phase}/avg_sim.nii",
    run:
        import numpy as np
        from nilearn.image import load_img
        template = load_img(input.maps[0])
        data = np.stack([load_img(m).get_fdata() for m in input.maps], axis=-1)
        mean = np.mean(data, axis=-1)
        new_img_like(template, mean, affine=template.affine).to_filename(output.avg_map)


rule find_significance_map:
    input:
        pos_maps=expand("results/searchlight_t_standard/{{feat}}-{{phase}}-{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
        neg_maps=expand("results/searchlight_t_standard/{{other_feat}}-{{phase}}-{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
        mask="results/searchlight_mask.nii.gz",
    params:
        permutations=config["sl_permutations"],
    resources:
        threads=config["searchlight_workers"],
    output:
        signed_pmap="results/searchlight_significance/{feat}-{other_feat}-{phase}/signed_pmap.nii",
        t_pmap="results/searchlight_significance/{feat}-{other_feat}-{phase}/t_pmap.nii",
        tmap="results/searchlight_significance/{feat}-{other_feat}-{phase}/tmap.nii",
    script:
        "../scripts/searchlight/searchlight_significance_map.py"


rule find_significance_map_single:
    input:
        maps=expand("results/searchlight_t_standard/{{feat}}-{{phase}}-{subject}.nii.gz",
            subject=config["fmri_subjects"],
        ),
        mask="results/searchlight_mask.nii.gz",
    params:
        permutations=config["sl_permutations"],
        two_sided_test=True
    resources:
        threads=config["searchlight_workers"],
    output:
        signed_pmap="results/searchlight_significance/{feat}-{phase}/signed_pmap.nii",
        t_pmap="results/searchlight_significance/{feat}-{phase}/t_pmap.nii",
        tmap="results/searchlight_significance/{feat}-{phase}/tmap.nii",
    script:
        "../scripts/searchlight/searchlight_significance_map_single.py"
