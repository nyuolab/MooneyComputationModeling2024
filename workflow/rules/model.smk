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
