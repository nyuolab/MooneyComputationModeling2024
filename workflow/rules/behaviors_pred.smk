rule make_diff_target:
    input:
        pre="data/behavior_{misc}s/{misc}_pre.mat",
        post="data/behavior_{misc}s/{misc}_post.mat",
    output:
        diff="data/behavior_{misc}s/{misc}_diff.mat",
    group: "behavior"
    script:
        "../scripts/predict_recognition/make_diff_target.py"


rule behavior_rec_pred:
    input:
        feature_source="results/behavior_trials/{pred_source}:{backbone}_seed{seed}.csv",
        features="results/behavior_trials/{pred_source}:{backbone}_seed{seed}_{feature_type}.npy",
        target_pred="data/behavior_corrects/correct_diff.mat",
    output:
        "results/behavior_predictions/diff:{pred_source}:subject{sub}:{backbone}_seed{seed}_recognize_{phase}_{feature_type}.csv",
    script:
        "../scripts/predict_recognition/predict_behavior_signals.py"