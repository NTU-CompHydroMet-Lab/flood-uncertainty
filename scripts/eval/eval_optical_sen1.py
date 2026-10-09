"""Evaluate the Sen1 optical water-only checkpoint on its test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.eval_sar import main

if __name__ == "__main__":
    main(default_config="configurations/edl_optical_sen1.json",
         default_output="artifacts/results/sen1_optical_test/metrics.json",
         default_plot=False)
