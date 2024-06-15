rule visualize_all_repetition_effect:
    input:
        model_records=expand(
            "results/long_native_trials/{synth_seed}_{backbone}_seed{seed}.csv",
            synth_seed=range(config["n_native_trials"]),
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),
    group: "plotting"
    output:
        plot="results/visualization/repetition_effect/+all_models.jpg",
    script:
        "../../scripts/visualization/plot_repetition.py"