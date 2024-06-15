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