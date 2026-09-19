"""
Master Replication Script: Forest Cover Type (P6)
=================================================
Course: CPE 513 — Artificial Neural Network
Author / Student: Level 550, Computer Engineering, FUT Minna

This script executes the entire empirical experimental pipeline from
the raw downloaded dataset (covtype.data.gz) through to report generation.

Execution sequence:
  1. Phase 1 & 2: Data audit and integrity checks (phase1_2_data_audit.py)
  2. Phase 3: Protocol freeze and baseline validation (phase3_protocol_and_tuning.py)
  3. Phase 4: Full 15-fold outer cross-validation matrix (phase4_full_experiment_matrix.py)
  4. Phase 5: Statistical analysis and publication figures (phase5_statistical_analysis_and_plots.py)
  5. Phase 6 & 7: Report generation in Word (.docx) and PDF (.pdf) (phase6_7_generate_report.py)
  6. Phase 8: Automated verification and parity checks (verify_reproducibility.py)

Usage:
  python run_all.py
  python run_all.py --fast     # Skips Phase 4 re-training if raw_fold_results.csv already exists
"""

import os
import sys
import io
import time
import subprocess

# Prevent Windows console encoding crashes and line buffering issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True, errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("Phase 1 & 2: Data Audit", "phase1_2_data_audit.py"),
    ("Phase 3: Protocol Freeze & Baselines", "phase3_protocol_and_tuning.py"),
    ("Phase 4: Full Experiment Matrix (15 outer folds)", "phase4_full_experiment_matrix.py"),
    ("Phase 5: Statistical Analysis & Figures", "phase5_statistical_analysis_and_plots.py"),
    ("Phase 6 & 7: Report Generation (DOCX & PDF)", "phase6_7_generate_report.py"),
    ("Phase 8: Verification & Parity Audit", "verify_reproducibility.py"),
]

def main():
    print("=" * 80)
    print("MASTER REPRODUCTION PIPELINE: FOREST COVER TYPE (P6)")
    print("=" * 80)
    print(f"Working directory: {BASE_DIR}")
    print(f"Python executable: {sys.executable}")
    
    fast_mode = "--fast" in sys.argv
    raw_results_exist = os.path.exists(os.path.join(BASE_DIR, "experiment_results", "raw_fold_results.csv"))
    
    t_global_start = time.time()
    
    for title, script_name in SCRIPTS:
        script_path = os.path.join(BASE_DIR, script_name)
        if not os.path.exists(script_path):
            print(f"ERROR: Script '{script_name}' not found at {script_path}!")
            sys.exit(1)
            
        # Optional fast skip for Phase 4 if raw results already archived and --fast is passed
        if script_name == "phase4_full_experiment_matrix.py" and fast_mode and raw_results_exist:
            print("\n" + "-" * 80)
            print(f"SKIPPING: {title} (--fast mode active, using archived raw fold results)")
            print("-" * 80)
            continue
            
        print("\n" + "-" * 80)
        print(f"STARTING: {title} ({script_name})")
        print("-" * 80)
        
        t0 = time.time()
        result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR, capture_output=False)
        t_elapsed = time.time() - t0
        
        if result.returncode != 0:
            print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}!")
            sys.exit(result.returncode)
            
        print(f"\n[COMPLETED] {title} in {t_elapsed:.1f}s")
        
    total_time = time.time() - t_global_start
    print("\n" + "=" * 80)
    print(f"ALL PHASES COMPLETED SUCCESSFULLY IN {total_time / 60:.2f} MINUTES")
    print("=" * 80)

if __name__ == "__main__":
    main()
