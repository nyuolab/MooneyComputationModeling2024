rule train_all_models:
    input:
        expand(
            "results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
            backbone=config["source_model"],
            seed=config["model_seeds"],
        ),


rule train_model:
    output:
        checkpoint="results/checkpoints/{backbone}_seed{seed}/best_model.ckpt",
    resources:
        nvidia_gpu=config["n_gpus"],
        threads=config["workers"],
    retries: 3
    shell:
        """
         python workflow/scripts/train_perceiver.py --checkpoint {output.checkpoint} \
        --config_file {config[configfile]} --seed {wildcards.seed} --backbone_name {wildcards.backbone}
        """
