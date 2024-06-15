rule get_gradient:
    input:
        error=expand(
            "results/prediction_error/counterfactual:post-post/subject{subject}.nii.gz",
            subject=config["fmri_subjects"]
        ),
        subject_responses=expand("data/fmri_image_order/S{subject}/response_order.xlsx",
            subject=config["fmri_subjects"]
        ),
        sample_fmri_sequence="results/fmri_sequences/human_subject4.csv",
    params:
        subjects=config["fmri_subjects"],
    output:
        grad_map=expand(
            "results/gradient_maps/counterfactual:post-post/gradient_{type}.nii.gz",
            type=["learned", "unlearned", "recognized"]
        ),
        explained_variance=expand(
            "results/gradient_maps/counterfactual:post-post/explained_variance_{type}.txt",
            type=["learned", "unlearned", "recognized"]
        ),
        masker="results/gradient_maps/counterfactual:post-post/masker.pkl",
        grad="results/gradient_maps/counterfactual:post-post/gradient_map.pkl",
    script:
        "../scripts/decomposition/get_gradient.py"
