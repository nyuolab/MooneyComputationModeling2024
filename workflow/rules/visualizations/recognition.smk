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