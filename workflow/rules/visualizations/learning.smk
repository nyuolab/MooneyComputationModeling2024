rule visualize_all_mooney_learning_long:
    input:
        model_records=expand(
            "results/long_native_trials/{synth_seed}_{backbone}_seed{seed}.csv",
            synth_seed=range(config["n_native_trials"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        plot="results/visualization/mooney_learning/+all_models_simple_long.jpg",
    group: "plotting"
    script:
        "../../scripts/visualization/plot_learning.py"


rule visualize_all_mooney_learning:
    input:
        model_records=expand(
            "results/behavior_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
            subject=range(config["n_behavior_subjects"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        plot="results/visualization/mooney_learning/+all_models.jpg",
    group: "plotting"
    script:
        "../../scripts/visualization/plot_learning.py"


rule make_long_behavior_test:
    input:
        checkpoint="results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
        sequence="results/long_native_sequence/{synth_seed}.csv",
    resources:
        nvidia_gpu=1
    retries: 10
    output:
        records="results/long_native_trials/{synth_seed}_{backbone}_seed{seed}.csv",
    script:
        "../../scripts/model_behaviors/model_behavior_long.py"


rule make_long_behavior_sequence:
    output:
        sequence="results/long_native_sequence/{synth_seed}.csv",
    script:
        "../../scripts/model_behaviors/make_native_sequence.py"


rule visualize_long_moooney_learning:
    input:
        model_records=expand(
            "results/long_native_trials/{synth_seed}_{backbone}_seed{seed}.csv",
            synth_seed=range(config["n_native_trials"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    output:
        plot="results/visualization/mooney_learning/+all_models_long.jpg",
    group: "plotting"
    script:
        "../../scripts/visualization/plot_learning_long.py"


rule compare_w_pilot:
    input:
        pre_subject_record="data/behavior_corrects/correct_pre.mat",
        post_subject_record="data/behavior_corrects/correct_post.mat",
        gray_subject_record="data/behavior_corrects/correct_gs.mat",
        model_record=expand(
            "results/behavior_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
            subject=range(config["n_behavior_subjects"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        )
    params:
        indicator=expand(
            "{subject};{backbone};{seed}",
            subject=range(config["n_behavior_subjects"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        )
    output:
        "results/learning_comparison.csv"
    script:
        "../../scripts/compare_learning.py"