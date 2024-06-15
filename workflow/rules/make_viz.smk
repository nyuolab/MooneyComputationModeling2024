
include: "visualizations/attention_map.smk"
include: "visualizations/learning.smk"
include: "visualizations/recognition.smk"
include: "visualizations/attention_shift.smk"
include: "visualizations/repetition_effect.smk"


"""
For figure 1 we want the 
"""


rule all_vis:
    input:
        # Comparison between ROIs
        "results/hypothesized_roi/comparisons.csv",
        "results/inter_component/comparisons.csv",

        # Comparison with standard ROIs
        "results/prediction_error_spatial_null/comparisons.csv",

        # Dimensionality and trustworthiness of representations
        "results/prediction_error_components/fisherS.csv",
        "results/prediction_error_components/boot_dimensions.csv",
        "results/prediction_error_components/trustworthiness.csv",

        # ROI level
        "results/encoding_roi_comparison/comparisons.csv",
        expand(
            "results/roi_based_encoding_similarity/{phase}/performance_detailed.csv",
            phase=["counterfactual-post"]
        ),

        # Pmap and thresholded differences
        expand(
            "results/encoding_significance/{phase}/pmap.nii",
            phase=["counterfactual-post"]
        ),
        expand(
            "results/encoding_significance/{phase}/thresholded_score_map.nii",
            phase=["counterfactual-post"]
        ),

        # Comparison with nc 
        expand(
            "results/{order}_nc_significance/{phase}-{phase}/pmap.nii",
            order=["above", "below"],
            phase=["post"]
        ),