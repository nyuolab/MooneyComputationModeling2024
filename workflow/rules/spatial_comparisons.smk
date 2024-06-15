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


eps = 1e-6
n_components = 4

from collections import OrderedDict
clumping_dict = OrderedDict(
    EVC=["v1", "v2", "v3", "hv4"],
    HLVC=["FG_L", "FG_R", "LO1", "LO2"],
    DMN=["DMN_MPFC", "DMN_PCC", "DMN_PL", "DMN_PR"],
    FPN=["FPN_FL", "FPN_FR", "FPN_PL", "FPN_PR"],
    Dorsal=["SPL"] + ["IPS%d" % idx for idx in range(6)],

    FG=["FG_L", "FG_R"],
    FP_F=["FPN_FL", "FPN_FR"],
    FP_P=["FPN_PL", "FPN_PR"],
    LOC=["LO1", "LO2"],
    SPL=["superior_parietal_lobule"],
    PO=["parietal_operculum"],
    PG=["postcentral_gyrus"],

    V12=["v1", "v2"],
    V34=["v3", "hv4"],

    MPFC=["DMN_MPFC"],
    PCC=["DMN_PCC"],
    DMN_PRL=["DMN_PL", "DMN_PR"],
)



rule get_pred_error:
    input:
        baseline_predictions="results/prediction_average/{control_phase}-{brain_phase}/human_subject{subject}.nii.gz",
        normal_predictions="results/prediction_average/{model_phase}-{brain_phase}/human_subject{subject}.nii.gz",
        groundtruth="results/fmris_standard/sub{subject}.nii.gz",
    output:
        error_ratio="results/prediction_error/{control_phase}:{model_phase}-{brain_phase}/subject{subject}.nii.gz"
    script:
        "../scripts/decomposition/get_relative_error.py"


rule make_clumped_rois:
    input:
        rois=expand(
            "data/beta_values/standard_brain_rois/{roi}.nii.gz",
            roi=config['standard_rois']
        )
    output:
        clumped_roi="results/clumped_rois/{c_roi}.nii.gz"
    params:
        rois=config['standard_rois']
    run:
        from nilearn.image import math_img, concat_imgs, resample_to_img, load_img
        curated = [img for idx, img in enumerate(input.rois) if params.rois[idx] in clumping_dict[wildcards.c_roi]]
        if len(curated) > 1:
            curated = [curated[0]] + [resample_to_img(img, curated[0]) for img in curated[1:]]
            math_img("np.clip(np.sum(imgs, axis=3), 0, 1)", imgs=concat_imgs(curated)).to_filename(output.clumped_roi)
        else:
            load_img(curated[0]).to_filename(output.clumped_roi)


rule get_thresholded_p_null:
    input:
        error="results/encoding_significance/counterfactual-{brain_phase}/thresholded_score_map.nii",
    output:
        null="results/encoding_roi_comparison/{brain_phase}/pmap.npy",
    run:
        import numpy as np
        from neuromaps.nulls import burt2020
        np.save(output.null, burt2020(input.error, atlas="mni152", n_perm=1000, n_proc=8, density="2mm"))


rule comparison_roi_thresholded_p:
    input:
        component="results/encoding_significance/counterfactual-{brain_phase}/thresholded_score_map.nii",
        null="results/encoding_roi_comparison/{brain_phase}/pmap.npy",
        roi="results/clumped_rois/{c_rois}.nii.gz"
    output:
        comparison="results/encoding_roi_comparison/comparisons_{brain_phase}/{c_rois}.npy",
    params:
        method="percent",
    script:
        "../scripts/decomposition/roi_component_comparison.py"


rule collect_all_pairs_thresholded:
    input:
        comparisons=expand(
            "results/encoding_roi_comparison/comparisons_post/{roi}.npy",
            roi=clumping_dict.keys(),
        )
    params:
        indicators=clumping_dict.keys()
    output:
        all_comparisons="results/encoding_roi_comparison/comparisons.csv"
    run:
        import numpy as np
        from pandas import DataFrame
        from collections import defaultdict

        container = defaultdict(list)
        for comparison, roi in zip(input.comparisons, params.indicators):
            similarity, pval  = np.load(comparison)
            container["ROI"].append(roi)
            container["Similarity"].append(similarity)
            container["p-value"].append(pval)

        DataFrame(container).to_csv(output.all_comparisons, index=False)
