"""Evaluation for FlowSat.

`evaluate_fmow` reproduces the FMoW-RGB numbers reported in the paper
(FID / CLIP / SSIM / LPIPS at N = 10,000). `eval_common` holds the metric
spine and the dataset-side plumbing it uses.

Both are imported lazily -- the metric stack (torchmetrics, torch-fidelity)
is an optional dependency, so importing `flowsat` does not require it:

    pip install -e ".[eval]"
    python -m flowsat.evaluation.evaluate_fmow --help
"""

__all__ = ["eval_common", "evaluate_fmow"]
