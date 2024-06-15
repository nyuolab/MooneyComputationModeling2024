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


explanation_rois = [
  "DMN_MPFC", "DMN_PCC", "DMN_PL", "DMN_PR",
  "FG_L", "FG_R", "FPN_FL", "FPN_FR", "FPN_PL", "FPN_PR",
  "LO1", "LO2", "v1", "v2", "v3", "hv4",
  "parietal_operculum", "superior_parietal_lobule"
]


hypotheses = [ 
    ["DMN_MPFC", "DMN_PCC", "DMN_PL", "DMN_PR"],
    ["v1", "v2", "v3", "hv4"],
    ["LO1", "LO2", "FG_L", "FG_R", "FPN_FL", "FPN_FR", "FPN_PL", "FPN_PR"],
    ["parietal_operculum", "superior_parietal_lobule", "postcentral_gyrus"],
]


rule compare_overlap_components:
    input:
        component="results/prediction_error_components/components_{idx1}.nii.gz",
        null="results/prediction_error_spatial_null/component_{idx1}.npy",
        roi="results/prediction_error_components/components_{idx2}.nii.gz",
    output:
        comparison="results/inter_component/comparisons/{idx1}-{idx2}.npy",
    params:
        method="dice",
    script:
        "../scripts/decomposition/roi_component_comparison.py"


rule collect_all_inter_component_pairs:
    input:
        comparisons=expand(
            "results/inter_component/comparisons/{idx1}-{idx2}.npy",
            idx1=range(n_components), idx2=range(n_components)
        )
    output:
        all_comparisons="results/inter_component/comparisons.csv"
    params:
        fromto=expand("{idx1}-{idx2}", idx1=range(n_components), idx2=range(n_components))
    run:
        import numpy as np
        from pandas import DataFrame
        from collections import defaultdict

        container = defaultdict(list)
        for comparison, indicator in zip(input.comparisons, params.fromto):
            from_, to_ = indicator.split("-")
            similarity, pval  = np.load(comparison)
            container["From component"].append(int(from_))
            container["To component"].append(int(to_))
            container["Similarity"].append(similarity)
            container["p-value"].append(pval)

        DataFrame(container).to_csv(output.all_comparisons, index=False)


rule make_hypothesized_rois:
    input:
        rois=lambda wildcards: expand(
            "data/beta_values/standard_brain_rois/{roi}.nii.gz",
            roi=hypotheses[int(wildcards.idx)]
        )
    output:
        clumped_roi="results/hypothesized_roi/{idx}.nii.gz"
    run:
        from nilearn.image import math_img, concat_imgs, resample_to_img
        curated = [input.rois[0]] + [resample_to_img(img, input.rois[0]) for img in input.rois[1:]]
        math_img("np.clip(np.sum(imgs, axis=3), 0, 1)", imgs=concat_imgs(curated)).to_filename(output.clumped_roi)


rule compare_hypothesized_roi:
    input:
        component="results/prediction_error_components/components_{idx}.nii.gz",
        null="results/prediction_error_spatial_null/component_{idx}.npy",
        roi="results/hypothesized_roi/{idx}.nii.gz"
    output:
        comparison="results/hypothesized_roi/comparisons/{idx}.npy",
    params:
        method="spearmanr",
    script:
        "../scripts/decomposition/roi_component_comparison.py"


rule collect_all_hypothesized_pairs:
    input:
        comparisons=expand("results/hypothesized_roi/comparisons/{idx}.npy", idx=range(n_components))
    output:
        all_comparisons="results/hypothesized_roi/comparisons.csv"
    run:
        import numpy as np
        from pandas import DataFrame
        from collections import defaultdict

        container = defaultdict(list)
        for index, comparison in enumerate(input.comparisons):
            similarity, pval  = np.load(comparison)
            container["Component"].append(index)
            container["Similarity"].append(similarity)
            container["p-value"].append(pval)

        DataFrame(container).to_csv(output.all_comparisons, index=False)


rule get_pred_error:
    input:
        baseline_predictions="results/prediction_average/{control_phase}-{brain_phase}/human_subject{subject}.nii.gz",
        normal_predictions="results/prediction_average/{model_phase}-{brain_phase}/human_subject{subject}.nii.gz",
        groundtruth="results/fmris_standard/sub{subject}.nii.gz",
    output:
        error_ratio="results/prediction_error/{control_phase}:{model_phase}-{brain_phase}/subject{subject}.nii.gz"
    script:
        "../scripts/decomposition/get_relative_error.py"


rule evaluate_n_components_single:
    input:
        error=expand(
            "results/prediction_error/counterfactual:post-post/subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        )
    output:
        record="results/prediction_error_components/fisherS.csv"
    params:
        boot=False
    script:
        "../scripts/decomposition/estimate_error_ID.py"


rule evaluate_n_components_single_boot:
    input:
        error=expand(
            "results/prediction_error/counterfactual:post-post/subject{subject}.nii.gz",
            subject=config["fmri_subjects"],
        )
    output:
        record="results/prediction_error_components/boot/{boot_idx}.csv"
    params:
        boot=True
    script:
        "../scripts/decomposition/estimate_error_ID.py"


rule evaluate_n_components_bootstrap:
    input:
        record=expand("results/prediction_error_components/boot/{boot_idx}.csv", boot_idx=range(1000))
    output:
        record="results/prediction_error_components/boot_dimensions.csv"
    run:
        import pandas as pd
        pd.concat([pd.read_csv(f) for f in input.record]).to_csv(output.record, index=False)


# Get the components of individual subject's scores
rule get_components:
    input:
        error=expand(
            "results/prediction_error/counterfactual:post-post/subject{subject}.nii.gz", subject=config["fmri_subjects"]
        )
    output:
        components=expand(
            "results/prediction_error_components/components_{idx}.nii.gz",
            idx=range(n_components)
        ),
        learner="results/prediction_error_components/learner.pkl",
        coefficients="results/prediction_error_components/record.csv",
        coefficients_raw="results/prediction_error_components/coef.npy",
        data="results/prediction_error_components/data.npy",
    script:
        "../scripts/decomposition/decompose_error.py"


rule trustworthiness:
    input:
        coef="results/prediction_error_components/coef.npy",
        original_data="results/prediction_error_components/data.npy",
    output:
        boot_trustworthiness="results/prediction_error_components/trustworthiness.csv",
    run:
        import pandas as pd
        import numpy as np
        from sklearn.utils import resample
        from sklearn.manifold import trustworthiness
        t = []
        X_reduced = np.load(input.coef)
        X = np.load(input.original_data)
        for _ in range(1000):
            X_reduced_, X_ = resample(X_reduced, X)
            t.append(trustworthiness(X_, X_reduced_, n_neighbors=5))
        pd.DataFrame(t, columns=["Trustworthiness"]).to_csv(output.boot_trustworthiness, index=False)


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


rule explain_component_with_roi:
    input:
        map_to_explain="results/prediction_error_components/components_{idx}.nii.gz",
        candidate_maps=expand(
            "data/beta_values/standard_brain_rois/{roi}.nii.gz", roi=explanation_rois
        ),
        null="results/prediction_error_spatial_null/component_{idx}.npy",
    params:
        roi_names=explanation_rois
    output:
        explanation="results/map_explaination/components/component_{idx}.csv"
    script:
        "../scripts/decomposition/lasso_roi.py"


rule collect_component_explanations:
    input:
        explanations=expand(
            "results/map_explaination/components/component_{idx}.csv",
            idx=range(n_components)
        )
    output:
        record="results/map_explaination/components.csv"
    run:
        import pandas as pd
        from collections import defaultdict
        container = defaultdict(list)
        for idx, explanation in enumerate(input.explanations):
            df = pd.read_csv(explanation)
            container["Component"].append(idx)
            for key in df.columns:
                container[key].append(df[key].values[0])
        pd.DataFrame(container).to_csv(output.record, index=False)


rule get_null:
    input:
        error="results/prediction_error_components/components_{idx}.nii.gz",
    output:
        null="results/prediction_error_spatial_null/component_{idx}.npy",
    run:
        import numpy as np
        from neuromaps.nulls import burt2020
        np.save(output.null, burt2020(input.error, atlas="mni152", n_perm=1000, n_proc=8, density="2mm"))


rule comparison_roi:
    input:
        component="results/prediction_error_components/components_{idx}.nii.gz",
        null="results/prediction_error_spatial_null/component_{idx}.npy",
        roi="results/clumped_rois/{c_rois}.nii.gz"
    output:
        comparison="results/prediction_error_spatial_null/comparisons/{c_rois}-{idx}.npy",
    params:
        method="coverage",
    script:
        "../scripts/decomposition/roi_component_comparison.py"


rule collect_all_pairs:
    input:
        comparisons=expand(
            "results/prediction_error_spatial_null/comparisons/{roi}-{idx}.npy",
            roi=clumping_dict.keys(), idx=range(n_components)
        )
    params:
        indicators=expand(
            "{roi};{idx}",
            roi=clumping_dict.keys(), idx=range(n_components)
        )
    output:
        all_comparisons="results/prediction_error_spatial_null/comparisons.csv"
    run:
        import numpy as np
        from pandas import DataFrame
        from collections import defaultdict

        container = defaultdict(list)
        for comparison, indicator in zip(input.comparisons, params.indicators):
            roi, index = indicator.split(";")
            similarity, pval  = np.load(comparison)
            container["ROI"].append(roi)
            container["Component"].append(index)
            container["Similarity"].append(similarity)
            container["p-value"].append(pval)

        DataFrame(container).to_csv(output.all_comparisons, index=False)


rule explain_score_with_roi:
    input:
        map_to_explain="results/encoding_significance/counterfactual-post/thresholded_score_map.nii",
        candidate_maps=expand(
            "data/beta_values/standard_brain_rois/{roi}.nii.gz", roi=explanation_rois
        ),
        null="results/encoding_roi_comparison/post/pmap.npy",
    params:
        roi_names=explanation_rois
    output:
        explanation="results/map_explaination/score_increase.csv"
    script:
        "../scripts/decomposition/lasso_roi.py"


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
