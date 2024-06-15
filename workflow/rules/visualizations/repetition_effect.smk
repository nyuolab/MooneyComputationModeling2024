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