"""
verify_setup.py
Run this after installing requirements.txt to confirm every library
needed for the satellite-climate-monitor project is correctly installed.

Usage:
    (venv) > python verify_setup.py
"""

import sys
import importlib


def check(module_name, display_name=None, version_attr="__version__"):
    display_name = display_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, version_attr, "unknown version")
        print(f"[OK]   {display_name:<20} -> {version}")
        return True
    except Exception as e:
        print(f"[FAIL] {display_name:<20} -> {type(e).__name__}: {e}")
        return False


def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version}\n")

    results = []

    # Core numerics
    results.append(check("numpy"))
    results.append(check("pandas"))
    results.append(check("matplotlib"))
    results.append(check("PIL", "Pillow", "__version__"))
    results.append(check("requests"))

    # Classical ML
    results.append(check("sklearn", "scikit-learn"))

    # Deep learning
    results.append(check("torch", "PyTorch"))
    results.append(check("torch_geometric", "PyTorch Geometric"))
    results.append(check("tensorflow", "TensorFlow"))

    # HuggingFace stack
    results.append(check("transformers", "HuggingFace Transformers"))
    results.append(check("datasets", "HuggingFace Datasets"))
    results.append(check("huggingface_hub", "HuggingFace Hub"))

    # Orchestration
    results.append(check("langchain", "LangChain"))
    results.append(check("langgraph", "LangGraph"))

    print("\n--- Functional checks ---")

    # Confirm torch can actually do a tensor op
    try:
        import torch
        x = torch.rand(3, 3)
        y = x @ x
        print(f"[OK]   torch tensor matmul -> shape {tuple(y.shape)}")
        print(f"       CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"[FAIL] torch tensor matmul -> {e}")
        results.append(False)

    # Confirm tensorflow can actually do a tensor op
    try:
        import tensorflow as tf
        a = tf.random.uniform((3, 3))
        b = tf.matmul(a, a)
        print(f"[OK]   tensorflow matmul -> shape {tuple(b.shape)}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"       GPU devices found: {len(gpus)}")
    except Exception as e:
        print(f"[FAIL] tensorflow matmul -> {e}")
        results.append(False)

    # Confirm sklearn can fit a trivial model
    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=5)
        clf.fit([[0, 0], [1, 1]], [0, 1])
        pred = clf.predict([[0.9, 0.9]])
        print(f"[OK]   sklearn RandomForest fit/predict -> {pred}")
    except Exception as e:
        print(f"[FAIL] sklearn RandomForest fit/predict -> {e}")
        results.append(False)

    print("\n--- Summary ---")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"{passed}/{total} import checks passed")

    if all(results):
        print("\nAll checks passed. Environment is ready for Step 2.")
    else:
        print("\nSome checks failed. Fix the [FAIL] lines above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()