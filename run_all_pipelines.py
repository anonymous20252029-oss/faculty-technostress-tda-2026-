"""Master Execution Script: End-to-End TDA Benchmarking Pipeline."""
import subprocess
import sys

def run_step(step_name, command):
    print(f"\n{'='*70}\n[RUNNING] {step_name}\n{'='*70}")
    res = subprocess.run([sys.executable] + command.split(), check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name}")

if __name__ == "__main__":
    run_step("1. Data Harmonization", "src/01_data_harmonization.py")
    run_step("2. Psychometric Validity Diagnostics", "src/02_psychometrics_validity.py")
    run_step("3. Topological Homology & Bootstrap Band", "src/03_topological_homology.py")
    run_step("4. Kepler-Mapper Complex Construction", "src/04_mapper_skeleton.py")
    run_step("5. Publication Figures Generation", "src/06_generate_figures.py")
    print("\n[SUCCESS] Entire research pipeline executed and all metrics validated.")
