rule quantify_attention_shift:
    input:
        mooney_images=[f"data/behavior_images/0_Mooney/{idx+1:04}.bmp" for idx in range(219)],
        grayscale_images=[f"data/behavior_images/1_grayscale/{idx+1001:04}.bmp" for idx in range(219)],
        model_checkpoint="results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
    output:
        token_idx="results/attention_shifts/idx_{backbone}_seed{seed}.txt",
        gray_score="results/attention_shifts/gray_{backbone}_seed{seed}.txt",
        gray_rank="results/attention_shifts/gray_rank_{backbone}_seed{seed}.txt",
        increase_score="results/attention_shifts/inc_{backbone}_seed{seed}.txt",
        pre_moran_score="results/attention_shifts/pre_moran_score_{backbone}_seed{seed}.txt",
        post_moran_score="results/attention_shifts/post_moran_score_{backbone}_seed{seed}.txt",
        correct="results/attention_shifts/correct_{backbone}_seed{seed}.txt",
        pre_correct="results/attention_shifts/pre_correct_{backbone}_seed{seed}.txt",

        pre_correlation="results/attention_shifts/pre_corr_score_{backbone}_seed{seed}.txt",
        post_correlation="results/attention_shifts/post_corr_score_{backbone}_seed{seed}.txt",
    params:
        mode="single"
    script:
        "../../scripts/visualization/attention_shift.py"


rule quantify_attention_shift_all:
    input:
        pre_correlation=expand(
            "results/attention_shifts/pre_corr_score_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
        post_correlation=expand(
            "results/attention_shifts/post_corr_score_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
        token_idx=expand(
            "results/attention_shifts/idx_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
        gray_score=expand(
            "results/attention_shifts/gray_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
        gray_rank=expand(
            "results/attention_shifts/gray_rank_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
        increase_score=expand(
            "results/attention_shifts/inc_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
        pre_moran_score=expand(
            "results/attention_shifts/pre_moran_score_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
        post_moran_score=expand(
            "results/attention_shifts/post_moran_score_{backbone}_seed{seed}.txt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        fig_out="results/visualization/attention_shifts/grayscale_phase_patch_sim_all_models.jpg",
        fig_out2="results/visualization/attention_shifts/increase_prob_all_models.jpg",
        fig_out3="results/visualization/attention_shifts/gray_hist_all_models.jpg",
        fig_out4="results/visualization/attention_shifts/inc_dec_compare_all_models.jpg",

        fig_out5="results/visualization/attention_shifts/change_magnitude_all_models.jpg",
        fig_out6="results/visualization/attention_shifts/attn_scattering_all_models.jpg",
    params:
        mode="all"
    script:
        "../../scripts/visualization/attention_shift.py"