from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Shared dataset location (kept outside this repo)
DEFAULT_DATA_ROOT = Path("/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data")

# Repo-local artifacts (not tracked by git)
DEFAULT_PRED_ROOT = PROJECT_ROOT / "artifacts" / "results" / "val_test_inference"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "artifacts" / "results" / "analysis_S2"
DEFAULT_FIGURE_ROOT = PROJECT_ROOT / "artifacts" / "results" / "figures"
