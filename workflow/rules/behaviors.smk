rule separate_human_behavior_sequences:
    input:
        image_order="data/behavior_image_order/img.mat",
        end_indicator="data/behavior_image_order/taskEndIndex.mat",
    group:
        "behavior"
    output:
        expand(
            "results/behavior_sequences/raw_human_sequence_{subject}.txt",
            subject=range(config["n_behavior_subjects"]),
        ),
    run:
        from scipy.io import loadmat

        img_order = loadmat(str(input.image_order))["img"]
        end_indicator = loadmat(str(input.end_indicator))["taskEndIndex"]
        img_name_order = [str(x).split("'")[1] for x in img_order[:, 0]]
        image_ends = [int(end) for end in end_indicator[:, 0]]

        for curr_subject in range(config["n_behavior_subjects"]):
            # Subset the order to contain only current subject
            start_idx = image_ends[curr_subject - 1] if curr_subject != 0 else 0
            end_idx = image_ends[curr_subject]
            img_name_order_curr = img_name_order[start_idx:end_idx]
            with open(str(output[curr_subject]), "w") as f:
                f.write("\n".join(img_name_order_curr))


rule parse_human_behavior_sequences_all:
    input:
        classes="data/behavior_classes.txt",
        image_order="results/behavior_sequences/raw_human_sequence_{subject}.txt",
    group:
        "behavior"
    output:
        "results/behavior_sequences/human_subject{subject}.csv",
    params:
        mode="behavior",
    script:
        "../scripts/parse_human_sequence.py"


rule behavior_trial_real:
    input:
        checkpoint="results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
        behavior_sequence="results/behavior_sequences/human_subject{subject}.csv",
    resources:
        nvidia_gpu=1,
    retries: 100
    output:
        behaviors="results/behavior_trials/human_subject{subject}:{backbone}_seed{seed}.csv",
        representations=temp(expand(
            "results/behavior_trials/human_subject{{subject}}:{{backbone}}_seed{{seed}}_{feature}.npy",
            feature=config["feature_types"]+config["layer_feature_types"]+config["static_feature_types"]+config["accumulate_feature_types"]
        ))
    script:
        "../scripts/model_behaviors/behavior_trial.py"
