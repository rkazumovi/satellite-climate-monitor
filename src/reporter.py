"""
reporter.py — Natural language climate monitoring report generation

Loads the real results from Stage 1 (classifier.py's classification_summary.json)
and Stage 2 (detector.py's change_detection_summary.csv), then uses a local
Hugging Face LLM (HuggingFaceTB/SmolLM2-1.7B-Instruct — a small model built for
on-device summarization) orchestrated through LangChain to write a natural
language report.

To guard against small-model inaccuracy: the LLM only narrates and interprets
the numbers, it never invents them. The final markdown report includes the
LLM's narrative PLUS a plain data appendix generated directly from the real
numbers (not the LLM), so nothing in the report's actual figures depends on
the model getting arithmetic right.

Run directly:
    python src/reporter.py
"""

import os
import json

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

CLASSIFICATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "classification_summary.json")
CHANGE_DETECTION_CSV_PATH = os.path.join(OUTPUT_DIR, "change_detection_summary.csv")


def load_classification_results():
    with open(CLASSIFICATION_SUMMARY_PATH, "r") as f:
        return json.load(f)


def load_change_detection_results():
    df = pd.read_csv(CHANGE_DETECTION_CSV_PATH)
    return df


def summarize_classification(results):
    per_class = results["per_class_metrics"]
    sorted_by_f1 = sorted(per_class.items(), key=lambda kv: kv[1]["f1_score"])
    weakest_class, weakest_metrics = sorted_by_f1[0]
    strongest_class, strongest_metrics = sorted_by_f1[-1]

    return {
        "best_model": results["best_model"],
        "test_accuracy": results["test_accuracy"],
        "macro_avg_f1": results["macro_avg_f1"],
        "num_test_samples": results["num_test_samples"],
        "weakest_class": weakest_class,
        "weakest_class_f1": weakest_metrics["f1_score"],
        "strongest_class": strongest_class,
        "strongest_class_f1": strongest_metrics["f1_score"],
    }


def summarize_change_detection(df):
    return {
        "num_pairs": len(df),
        "mean_precision": df["precision"].mean(),
        "mean_recall": df["recall"].mean(),
        "mean_f1": df["f1"].mean(),
        "mean_accuracy": df["accuracy"].mean(),
        "mean_pct_true_change": df["pct_pixels_changed_true"].mean(),
        "pairs_with_vegetation_loss": int((df["mean_ndvi_change"] < 0).sum()),
        "worst_vegetation_loss_pair": df.loc[df["mean_ndvi_change"].idxmin(), "pair_id"],
        "worst_vegetation_loss_value": df["mean_ndvi_change"].min(),
        "most_changed_pair": df.loc[df["pct_pixels_changed_true"].idxmax(), "pair_id"],
        "most_changed_pct": df["pct_pixels_changed_true"].max(),
    }


def build_data_summary_text(classification_summary, change_summary):
    """Plain-text block of the real numbers, handed to the LLM as its only source of facts."""
    return f"""
LAND USE CLASSIFICATION RESULTS (EuroSAT satellite imagery, {classification_summary['num_test_samples']} test images):
- Best performing model: {classification_summary['best_model']}
- Overall test accuracy: {classification_summary['test_accuracy']:.1%}
- Macro-averaged F1 score: {classification_summary['macro_avg_f1']:.3f}
- Strongest class: {classification_summary['strongest_class']} (F1 = {classification_summary['strongest_class_f1']:.3f})
- Weakest class: {classification_summary['weakest_class']} (F1 = {classification_summary['weakest_class_f1']:.3f})

CHANGE DETECTION RESULTS ({change_summary['num_pairs']} real Sentinel-2 before/after image pairs, OSCD dataset):
- Mean precision: {change_summary['mean_precision']:.3f}
- Mean recall: {change_summary['mean_recall']:.3f}
- Mean F1 score: {change_summary['mean_f1']:.3f}
- Mean pixel-level accuracy: {change_summary['mean_accuracy']:.1%}
- Average ground-truth change coverage: {change_summary['mean_pct_true_change']:.2f}% of pixels
- Pairs showing net vegetation loss (NDVI decline): {change_summary['pairs_with_vegetation_loss']} of {change_summary['num_pairs']}
- Most severe vegetation loss: pair #{change_summary['worst_vegetation_loss_pair']} (mean NDVI change = {change_summary['worst_vegetation_loss_value']:.4f})
- Largest observed change area: pair #{change_summary['most_changed_pair']} ({change_summary['most_changed_pct']:.2f}% of pixels changed)
""".strip()


def build_llm():
    print(f"Loading LLM: {MODEL_ID} (first run downloads ~3.5GB, then cached)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def generate_report_narrative(chat_model, data_summary_text):
    system_prompt = (
        "You are a climate monitoring analyst writing a concise report section for a "
        "satellite-based land use and environmental change monitoring system. You will "
        "be given real, already-computed statistics. Write a clear, factual narrative "
        "interpreting these numbers for a non-technical stakeholder. Use ONLY the numbers "
        "given to you — never invent or estimate additional figures. Structure your "
        "response with these sections: 'Land Use Classification Summary', 'Change "
        "Detection Summary', and 'Overall Assessment'. Keep it under 400 words."
    )

    human_prompt = f"Here are the results to report on:\n\n{data_summary_text}\n\nWrite the report."

    print("\nGenerating report narrative with the LLM (this may take a minute or two on CPU)...")
    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    return response.content


def save_report(narrative_text, data_summary_text):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "climate_monitoring_report.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Satellite Climate Monitoring Report\n\n")
        f.write(narrative_text.strip())
        f.write("\n\n---\n\n")
        f.write("## Data Appendix (raw computed statistics)\n\n")
        f.write("```\n")
        f.write(data_summary_text)
        f.write("\n```\n")

    print(f"\nSaved report: {path}")
    return path


def main():
    print("Loading Stage 1 and Stage 2 results...")
    classification_results = load_classification_results()
    change_detection_df = load_change_detection_results()

    classification_summary = summarize_classification(classification_results)
    change_summary = summarize_change_detection(change_detection_df)

    data_summary_text = build_data_summary_text(classification_summary, change_summary)
    print("\n--- Data summary handed to the LLM ---")
    print(data_summary_text)

    chat_model = build_llm()
    narrative_text = generate_report_narrative(chat_model, data_summary_text)

    print("\n--- Generated narrative ---")
    print(narrative_text)

    save_report(narrative_text, data_summary_text)

    print("\nStep 5 (reporter.py) complete.")


if __name__ == "__main__":
    main()