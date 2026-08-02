"""
main.py — Full pipeline entry point

Runs all six stages of the satellite climate monitoring pipeline in order:
  1. data.py       — EuroSAT download + train/test/validation split
  2. features.py   — CNN (ResNet18) feature extraction
  3. classifier.py — scikit-learn land use classification
  4. detector.py   — OSCD real change detection
  5. reporter.py   — LLM-generated natural language report
  6. visualize.py  — summary charts

Each stage is skipped automatically if its output already exists, to avoid
re-downloading or re-training unnecessarily. Use --force to rerun everything,
or --only to run a specific stage (or comma-separated list of stages).

Usage:
    python main.py                        # run everything, skip completed stages
    python main.py --force                # rerun everything regardless
    python main.py --only classifier       # run just one stage
    python main.py --only detector,reporter  # run a subset
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Each stage: (module name, human label, path used to detect "already done")
STAGES = [
    ("data", "Stage 1a: Data download + split",
     os.path.join(OUTPUT_DIR, "sample_images")),
    ("features", "Stage 1b: CNN feature extraction",
     os.path.join(OUTPUT_DIR, "features", "train_features.npy")),
    ("classifier", "Stage 1c: Land use classification",
     os.path.join(OUTPUT_DIR, "classification_summary.json")),
    ("detector", "Stage 2: Change detection",
     os.path.join(OUTPUT_DIR, "change_detection_summary.csv")),
    ("reporter", "Stage 3: LLM report generation",
     os.path.join(OUTPUT_DIR, "reports", "climate_monitoring_report.md")),
    ("visualize", "Stage 4: Summary visualizations",
     os.path.join(OUTPUT_DIR, "visualizations", "summary_dashboard.png")),
]


def is_done(marker_path):
    return os.path.exists(marker_path)


def run_stage(module_name, label, marker_path, force):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    if not force and is_done(marker_path):
        print(f"Already completed (found {marker_path}). Skipping.")
        print("Use --force to rerun this stage anyway.")
        return True

    try:
        module = __import__(module_name)
    except ImportError as e:
        print(f"FAILED to import '{module_name}': {e}")
        return False

    start = time.time()
    try:
        module.main()
    except Exception as e:
        print(f"FAILED during '{module_name}.main()': {type(e).__name__}: {e}")
        return False

    elapsed = time.time() - start
    print(f"\n{label} finished in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the satellite climate monitoring pipeline")
    parser.add_argument("--force", action="store_true",
                         help="Rerun all stages even if outputs already exist")
    parser.add_argument("--only", type=str, default=None,
                         help="Comma-separated stage name(s) to run, e.g. 'classifier' or 'detector,reporter'")
    args = parser.parse_args()

    stages_to_run = STAGES
    if args.only:
        requested = {name.strip() for name in args.only.split(",")}
        valid_names = {s[0] for s in STAGES}
        unknown = requested - valid_names
        if unknown:
            print(f"Unknown stage name(s): {unknown}. Valid options: {sorted(valid_names)}")
            sys.exit(1)
        stages_to_run = [s for s in STAGES if s[0] in requested]

    print("Satellite Climate Monitoring Pipeline")
    print(f"Running {len(stages_to_run)} stage(s): {[s[0] for s in stages_to_run]}")

    pipeline_start = time.time()
    results = {}

    for module_name, label, marker_path in stages_to_run:
        success = run_stage(module_name, label, marker_path, args.force)
        results[module_name] = success
        if not success:
            print(f"\nStopping pipeline: '{module_name}' failed.")
            break

    total_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for module_name, _, _ in stages_to_run:
        status = "OK" if results.get(module_name) else ("FAILED" if module_name in results else "NOT RUN")
        print(f"  {module_name:<12} {status}")
    print(f"\nTotal time: {total_elapsed:.1f}s")

    if all(results.values()):
        print("\nPipeline complete. See outputs/ for all results, reports, and visualizations.")
    else:
        print("\nPipeline did not complete successfully. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()