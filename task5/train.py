"""
MyBot NLU — Training Script  (v3 — Per-Workflow)
==================================================

Reads training data from ./training_data/<workflow>.json
and trains the intent classifier via the NLU service.

Usage:
  python train.py --workflow welcome-intent     # train one workflow
  python train.py --all                         # train ALL workflows in training_data/
  python train.py --list                        # list all available training files + status
  python train.py --status welcome-intent       # check one workflow's trained model status
  python train.py --reset welcome-intent        # delete a specific trained model
  python train.py --augment 5 --workflow query-management  # custom augmentation
  python train.py --test "check my application" --workflow applications-status

Requirements:
  - NLU service must be running: start.bat
  - Service URL: http://localhost:8001
"""

import sys
import json
import argparse
import time
import requests
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
NLU_URL           = "http://127.0.0.1:8000"
TRAINING_DATA_DIR = Path(__file__).parent / "training_data"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step: str):
    print(f"\n  -> {step}")


def check_service() -> dict:
    """Check if the NLU service is up and base models are loaded."""
    print_step("Checking NLU service ...")
    try:
        r = requests.get(f"{NLU_URL}/health", timeout=15)
        data = r.json()
        if data.get("status") != "ok":
            print(f"  [X] Service not ready: {data}")
            sys.exit(1)
        if not data.get("modelsLoaded"):
            print("  [WAIT] Models still loading. Waiting 10 seconds ...")
            time.sleep(10)
            return check_service()
        trained = data.get("totalTrainedWorkflows", 0)
        print(f"  [OK] Service ready  |  Trained workflows: {trained}")
        return data
    except requests.exceptions.ConnectionError:
        print(f"\n  [X] Cannot connect to NLU service at {NLU_URL}")
        print("    Start the service first:  start.bat")
        sys.exit(1)


def get_all_training_files() -> list[Path]:
    """Return all .json files in the training_data/ directory."""
    if not TRAINING_DATA_DIR.exists():
        print(f"  [X] training_data/ directory not found at {TRAINING_DATA_DIR}")
        sys.exit(1)
    files = sorted(TRAINING_DATA_DIR.glob("*.json"))
    # Exclude files starting with _ (e.g. _template.json)
    return [f for f in files if not f.name.startswith("_")]


def load_training_file(filepath: Path) -> dict:
    """Load and validate a training data JSON file."""
    if not filepath.exists():
        print(f"  [X] File not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("subIntents"):
        print(f"  [X] No subIntents found in: {filepath.name}")
        sys.exit(1)
    if not data.get("workflowId"):
        print(f"  [X] No workflowId found in: {filepath.name}")
        sys.exit(1)
    return data


def print_training_summary(data: dict):
    """Print a table of subintents + example counts."""
    wf = data.get("workflowId", "?")
    print(f"\n  WorkflowId: {wf}")
    print(f"  {'#':<4} {'SubIntent':<35} {'Examples':>10}")
    print("  " + "-" * 52)
    for i, sub in enumerate(data["subIntents"], 1):
        count = len(sub.get("examples", []))
        print(f"  {i:<4} {sub['name']:<35} {count:>10}")
    total = sum(len(s.get("examples", [])) for s in data["subIntents"])
    print("  " + "-" * 52)
    print(f"  {'TOTAL':<39} {total:>10} examples")


def train_workflow(data: dict, augment: int):
    """Send a training request for one workflow to the NLU service."""
    workflow_id = data["workflowId"]
    payload = {
        "workflowId":         workflow_id,
        "subIntents":         data["subIntents"],
        "augmentationFactor": augment,
    }

    print_step(f"Training '{workflow_id}' (augmentationFactor={augment}) ...")
    print("  This may take 30–90 seconds ...")
    t0 = time.time()

    try:
        r = requests.post(
            f"{NLU_URL}/train",
            json=payload,
            timeout=300,
        )
        elapsed = round(time.time() - t0, 1)

        if r.status_code != 200:
            print(f"\n  [X] Training failed [{r.status_code}]: {r.text}")
            return False

        result = r.json()
        meta   = result.get("training", {})

        print(f"\n  [OK] Done in {elapsed}s!")
        print(f"  {'-' * 50}")
        print(f"  Trained at:    {meta.get('trainedAt', '?')}")
        print(f"  Sentences:     {meta.get('totalSentences', '?')}")
        print(f"  Intents:       {meta.get('totalIntents', '?')}")
        print(f"  Augmentation:  {meta.get('augmentationFactor', '?')}x")

        cv = meta.get("cvAccuracy")
        if cv is not None:
            bar = "*" * int(cv * 20) + "-" * (20 - int(cv * 20))
            print(f"  CV Accuracy:   {cv:.2%}  [{bar}]")
        else:
            print("  CV Accuracy:   (insufficient samples)")

        print(f"\n  Subintents trained:")
        for intent in meta.get("intents", []):
            count = meta.get("exampleCounts", {}).get(intent, "?")
            print(f"    [OK]  {intent}  ({count} augmented examples)")

        return True

    except requests.exceptions.Timeout:
        print(f"\n  [X] Training timed out after 5 minutes.")
        return False


def check_status(workflow_id: str = None):
    """Check training status for one or all workflows."""
    try:
        url = f"{NLU_URL}/training-status"
        if workflow_id:
            url += f"?workflowId={workflow_id}"
        r = requests.get(url, timeout=5)
        data = r.json()

        if workflow_id:
            if data.get("trained"):
                meta = data.get("metadata", {})
                print(f"\n  [OK] '{workflow_id}' has a trained model")
                print(f"    Trained at : {meta.get('trainedAt', '?')}")
                print(f"    Intents    : {meta.get('totalIntents', '?')}")
                print(f"    Sentences  : {meta.get('totalSentences', '?')}")
                cv = meta.get("cvAccuracy")
                if cv:
                    print(f"    CV Accuracy: {cv:.2%}")
                for intent in meta.get("intents", []):
                    print(f"    * {intent}")
            else:
                print(f"\n  [X] '{workflow_id}' has no trained model — using zero-shot mode")
        else:
            total = data.get("totalTrained", 0)
            workflows = data.get("workflows", {})
            print(f"\n  Total trained workflows: {total}")
            print(f"\n  {'WorkflowId':<40} {'Intents':>8} {'CV Accuracy':>12} {'Trained At'}")
            print("  " + "-" * 85)
            for wf_id, info in workflows.items():
                meta      = info.get("metadata", {})
                n_intents = meta.get("totalIntents", "?")
                cv        = meta.get("cvAccuracy")
                cv_str    = f"{cv:.2%}" if cv else "N/A"
                trained_at = meta.get("trainedAt", "?")[:19] if meta.get("trainedAt") else "?"
                print(f"  {wf_id:<40} {str(n_intents):>8} {cv_str:>12} {trained_at}")

            # Also show which training files don't have trained models yet
            all_files = get_all_training_files()
            untrained = [f.stem for f in all_files if f.stem not in workflows]
            if untrained:
                print(f"\n  Not yet trained ({len(untrained)}):")
                for u in untrained:
                    print(f"    [X] {u}")

    except Exception as e:
        print(f"  [X] Could not check status: {e}")


def reset_model(workflow_id: str):
    """Delete a specific trained model."""
    confirm = input(f"\n  Delete trained model for '{workflow_id}'? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("  Cancelled.")
        return
    try:
        r = requests.delete(f"{NLU_URL}/training-status?workflowId={workflow_id}", timeout=5)
        print(f"  [OK] {r.json().get('message', 'Done')}")
    except Exception as e:
        print(f"  [X] Failed: {e}")


def test_prediction(text: str, workflow_id: str):
    """Quick test: run /analyze for a given text + workflowId."""
    try:
        r = requests.post(
            f"{NLU_URL}/analyze",
            json={
                "text":                text,
                "workflowId":          workflow_id,
                "subIntents":          [],
                "confidenceThreshold": 0.0,
            },
            timeout=10,
        )
        data = r.json()
        print(f"\n  Input      : \"{text}\"")
        print(f"  WorkflowId : {workflow_id}")
        print(f"  Intent     : {data.get('intent', '?')}")
        print(f"  Mode       : {data.get('intentMode', '?')}")
        print(f"  Confidence : {data.get('confidence', 0):.2%}")
        entities = data.get("entities", {})
        if entities:
            print("\n  Entities:")
            for k, v in entities.items():
                print(f"    * {k}: {v}")
        print(f"\n  Top scores:")
        for s in data.get("allScores", [])[:5]:
            bar = "*" * int(s["score"] * 20)
            print(f"    {s['name']:<35} {s['score']:.4f}  {bar}")
    except Exception as e:
        print(f"  [X] Prediction failed: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global NLU_URL

    parser = argparse.ArgumentParser(
        description="MyBot NLU Training Script v3 — per-workflow training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--workflow", type=str, default=None,
                        help="Train a specific workflow (e.g. welcome-intent, query-management)")
    parser.add_argument("--all",      action="store_true",
                        help="Train ALL workflows in training_data/ folder")
    parser.add_argument("--list",     action="store_true",
                        help="List all available training files and their trained status")
    parser.add_argument("--status",   type=str, nargs="?", const="__all__",
                        help="Check training status. Optionally specify workflowId.")
    parser.add_argument("--reset",    type=str, default=None,
                        help="Delete trained model for a specific workflowId")
    parser.add_argument("--augment",  type=int, default=3,
                        help="Augmentation factor per example (default: 3)")
    parser.add_argument("--test",     type=str, default=None,
                        help="Test a prediction after training (requires --workflow)")
    parser.add_argument("--url",      default=NLU_URL,
                        help=f"NLU service URL (default: {NLU_URL})")
    args = parser.parse_args()

    NLU_URL = args.url

    print_banner("MyBot NLU Training Script v3")

    # ── List available training files ─────────────────────────────────────────
    if args.list:
        check_service()
        check_status()
        return

    # ── Check status ──────────────────────────────────────────────────────────
    if args.status is not None:
        check_service()
        wf = None if args.status == "__all__" else args.status
        check_status(wf)
        return

    # ── Reset/delete a model ──────────────────────────────────────────────────
    if args.reset:
        check_service()
        reset_model(args.reset)
        return

    # ── Train ONE workflow ────────────────────────────────────────────────────
    if args.workflow:
        check_service()
        filepath = TRAINING_DATA_DIR / f"{args.workflow}.json"
        data = load_training_file(filepath)
        print_training_summary(data)

        print(f"\n  augmentationFactor = {args.augment}")
        confirm = input("  Start training? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("  Cancelled.")
            return

        success = train_workflow(data, args.augment)
        if success and args.test:
            print_step("Running test prediction ...")
            test_prediction(args.test, data["workflowId"])

        if success:
            print_banner(f"Done! '{args.workflow}' model trained and ready.")

        return

    # ── Train ALL workflows ───────────────────────────────────────────────────
    if args.all:
        check_service()
        files = get_all_training_files()
        if not files:
            print(f"  [X] No training files found in {TRAINING_DATA_DIR}")
            sys.exit(1)

        print(f"\n  Found {len(files)} training file(s) in training_data/:")
        for f in files:
            print(f"    * {f.name}")

        confirm = input(f"\n  Train ALL {len(files)} workflows? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("  Cancelled.")
            return

        results = {}
        for filepath in files:
            data = load_training_file(filepath)
            print(f"\n{'-' * 60}")
            print_training_summary(data)
            success = train_workflow(data, args.augment)
            results[data["workflowId"]] = "[OK]" if success else "[X] FAILED"

        print_banner("All Workflows Training Complete")
        print("\n  Results:")
        for wf_id, status in results.items():
            print(f"    {status}  {wf_id}")
        return

    # ── No args — show help ───────────────────────────────────────────────────
    parser.print_help()
    print("""
  Examples:
    python train.py --list
    python train.py --workflow welcome-intent
    python train.py --workflow query-management --augment 5
    python train.py --all
    python train.py --status
    python train.py --status welcome-intent
    python train.py --test "check my application status" --workflow applications-status
    python train.py --reset query-management
""")


if __name__ == "__main__":
    main()
