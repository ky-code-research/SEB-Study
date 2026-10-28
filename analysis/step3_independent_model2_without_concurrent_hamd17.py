import step2_fourlevel_no_concurrent_hamd17_education3 as model
from step3_independent_model_core import ROOT, run_independent


OUT = ROOT / "outputs" / "step3_independent_model2_without_hamd17_2026-10-09"
SPEC = {
    "rhs": model.RHS,
    "expected_parameters": 50,
    "expected_g": 24,
    "expected_x": 23,
    "step2_parameters": ROOT / "outputs" / "step2_fourlevel_no_concurrent_hamd17_education3" / "09_selected_parameter_estimates.csv",
    "concurrent_hamd17": False,
}


if __name__ == "__main__":
    run_independent("Model 2", SPEC, OUT)


