"""
app.py

[Extension, Demo] Interactive Gradio demonstration of the project's adaptive
reliability-aware uncertainty estimation. Reads entirely from demo_cache.json
(produced by build_demo_cache.py) -- no PyTorch, no GPU, no model loading at runtime.

Two tabs:
    1. Inspect a Sample   -- pick a random real test image, see both methods'
                             predictions, uncertainty, and accept/review decision
                             side by side.
    2. Threshold Explorer -- an interactive rejection-threshold slider showing the
                             AAC-style accuracy/throughput trade-off live, for both
                             Static CAE and the project's Fully Combined method.

Run from the project root:
    python demo/app.py
"""

import json
import os
import random

import gradio as gr
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, "demo", "demo_cache.json")

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    CACHE = json.load(f)

IMAGES_DIR = os.path.join(PROJECT_ROOT, CACHE["images_dir_relative"])
SAMPLES = CACHE["samples"]
CURVE = CACHE["rejection_curve"]
SUMMARY = CACHE["summary"]

DEFAULT_THRESHOLD = 15  # % of most-uncertain samples flagged for review, by default


# ---------------------------------------------------------------------------
# Tab 1: Inspect a Sample
# ---------------------------------------------------------------------------

def format_decision(rank_pct: float, threshold: float) -> str:
    if rank_pct < threshold:
        return "\U0001F6A9 Flagged for human review"
    return "\u2705 Auto-accepted"


def pick_random_sample(threshold: float):
    sample = random.choice(SAMPLES)
    return render_sample(sample, threshold)


def render_sample(sample: dict, threshold: float):
    image_path = os.path.join(IMAGES_DIR, sample["filename"])
    if not os.path.exists(image_path):
        image_path = None  # Gradio shows an empty image box rather than erroring

    true_label = "Positive reaction" if sample["hard_label"] == 1 else "Negative reaction"
    pred_label = "Positive reaction" if sample["pred_prob"] >= 0.5 else "Negative reaction"
    correctness = "\u2705 Correct" if sample["correct"] else "\u274C Incorrect"

    info_md = (
        f"**Filename:** `{sample['filename']}`  \n"
        f"**Reagent:** {sample['reagent']}  \n"
        f"**True label:** {true_label}  \n"
        f"**Model prediction:** {pred_label} (confidence: {max(sample['pred_prob'], 1 - sample['pred_prob']):.3f})  \n"
        f"**Prediction correctness:** {correctness}"
    )

    static_md = (
        f"### Static CAE (base paper's method)\n"
        f"**Uncertainty score:** {sample['static_uncertainty']:.5f}  \n"
        f"**Percentile rank:** {sample['static_rank_pct']:.1f}% "
        f"(0% = most uncertain sample in the test set, 100% = least uncertain)  \n"
        f"**Decision at {threshold:.0f}% threshold:** {format_decision(sample['static_rank_pct'], threshold)}"
    )

    combined_md = (
        f"### Fully Combined (this project's method)\n"
        f"**Uncertainty score:** {sample['combined_uncertainty']:.5f}  \n"
        f"**Percentile rank:** {sample['combined_rank_pct']:.1f}%  \n"
        f"**Decision at {threshold:.0f}% threshold:** {format_decision(sample['combined_rank_pct'], threshold)}"
    )

    return image_path, info_md, static_md, combined_md


def update_decisions_only(sample_state: dict, threshold: float):
    if sample_state is None:
        return gr.update(), gr.update()
    static_md = (
        f"### Static CAE (base paper's method)\n"
        f"**Uncertainty score:** {sample_state['static_uncertainty']:.5f}  \n"
        f"**Percentile rank:** {sample_state['static_rank_pct']:.1f}%  \n"
        f"**Decision at {threshold:.0f}% threshold:** "
        f"{format_decision(sample_state['static_rank_pct'], threshold)}"
    )
    combined_md = (
        f"### Fully Combined (this project's method)\n"
        f"**Uncertainty score:** {sample_state['combined_uncertainty']:.5f}  \n"
        f"**Percentile rank:** {sample_state['combined_rank_pct']:.1f}%  \n"
        f"**Decision at {threshold:.0f}% threshold:** "
        f"{format_decision(sample_state['combined_rank_pct'], threshold)}"
    )
    return static_md, combined_md


def pick_and_store(threshold: float):
    sample = random.choice(SAMPLES)
    image_path, info_md, static_md, combined_md = render_sample(sample, threshold)
    return image_path, info_md, static_md, combined_md, sample


# ---------------------------------------------------------------------------
# Tab 2: Threshold Explorer
# ---------------------------------------------------------------------------

def make_curve_plot(threshold: float):
    thresholds = CURVE["thresholds_pct"]
    static_acc = CURVE["static_accuracy_pct"]
    combined_acc = CURVE["combined_accuracy_pct"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, static_acc, "--", color="#1F2D3D", label="Static CAE", linewidth=2)
    ax.plot(thresholds, combined_acc, "-", color="#D9672A", label="Fully Combined", linewidth=2)
    ax.axvline(x=threshold, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Rejected data part (%)")
    ax.set_ylabel("Accuracy on remaining samples (%)")
    ax.set_title("Accuracy vs. Rejection Threshold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def update_threshold_explorer(threshold: float):
    idx = int(round(threshold))
    idx = max(0, min(idx, len(CURVE["thresholds_pct"]) - 1))

    static_acc_at_t = CURVE["static_accuracy_pct"][idx]
    combined_acc_at_t = CURVE["combined_accuracy_pct"][idx]

    summary_md = (
        f"**At a {threshold:.0f}% rejection threshold:**\n\n"
        f"| | Static CAE | Fully Combined |\n"
        f"|---|---|---|\n"
        f"| Accuracy on retained samples | {static_acc_at_t:.2f}% | {combined_acc_at_t:.2f}% |\n"
        f"| Samples flagged for review | {threshold:.0f}% of {CACHE['n_test']} "
        f"(\u2248{int(threshold / 100 * CACHE['n_test'])} images) | same |\n\n"
        f"**Overall AAC (lower is better):** Static CAE = {SUMMARY['static']['aac_x1e4']:.2f}, "
        f"Fully Combined = {SUMMARY['combined']['aac_x1e4']:.2f}  \n"
        f"**Overall AUROC (higher is better):** Static CAE = {SUMMARY['static']['auroc']:.4f}, "
        f"Fully Combined = {SUMMARY['combined']['auroc']:.4f}"
    )

    return make_curve_plot(threshold), summary_md


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------

with gr.Blocks(title="Adaptive Reliability-Aware Confidence Learning \u2014 Demo") as demo:
    gr.Markdown(
        "# Adaptive Reliability-Aware Confidence Learning\n"
        "### Interactive demonstration \u2014 BloodyWell blood-typing test set "
        f"({CACHE['n_test']} images, overall accuracy {CACHE['overall_accuracy']:.2%})\n\n"
        "This demo compares the base paper's **Static CAE** method against this project's "
        "**Fully Combined** method (model-level + expert-level reliability weighting), "
        "using real, held-out test images the models have never trained on."
    )

    with gr.Tabs():
        with gr.TabItem("Inspect a Sample"):
            gr.Markdown(
                "Click the button to load a random real test image and compare how each "
                "method evaluates it. Adjust the threshold slider to see when each method "
                "would flag this specific sample for human review."
            )
            sample_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    image_display = gr.Image(label="Test Sample", type="filepath", height=300)
                    sample_button = gr.Button("\U0001F3B2 Show Random Sample", variant="primary")
                    info_display = gr.Markdown()
                with gr.Column(scale=1):
                    threshold_slider_1 = gr.Slider(
                        minimum=0, maximum=99, value=DEFAULT_THRESHOLD, step=1,
                        label="Rejection threshold (% most uncertain samples flagged)",
                    )
                    static_display = gr.Markdown()
                    combined_display = gr.Markdown()

            sample_button.click(
                fn=pick_and_store,
                inputs=[threshold_slider_1],
                outputs=[image_display, info_display, static_display, combined_display, sample_state],
            )
            threshold_slider_1.change(
                fn=update_decisions_only,
                inputs=[sample_state, threshold_slider_1],
                outputs=[static_display, combined_display],
            )

        with gr.TabItem("Threshold Explorer"):
            gr.Markdown(
                "Drag the slider to set what fraction of the most uncertain test samples "
                "would be automatically flagged for human review. Watch how quickly "
                "accuracy on the remaining, auto-accepted samples climbs for each method."
            )
            threshold_slider_2 = gr.Slider(
                minimum=0, maximum=99, value=DEFAULT_THRESHOLD, step=1,
                label="Rejection threshold (%)",
            )
            with gr.Row():
                curve_plot = gr.Plot(label="Accuracy-Rejection Curve")
                explorer_summary = gr.Markdown()

            threshold_slider_2.change(
                fn=update_threshold_explorer,
                inputs=[threshold_slider_2],
                outputs=[curve_plot, explorer_summary],
            )
            demo.load(
                fn=update_threshold_explorer,
                inputs=[threshold_slider_2],
                outputs=[curve_plot, explorer_summary],
            )


if __name__ == "__main__":
    demo.launch()