import json

import matplotlib.pyplot as plt
from venn import venn

# --- Load data ---
data_file = (
    "localization_results/augmented_retrieval_result_gpt-oss_seperated_query_t0.7.json"
)
base_data_file = "localization_results/retrieval_results_code_test_with_path_k50_20lines_with_patch_file.json"
agentic_file = "localization_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json"
agentic_with_base_only_file = (
    "localization_results/agentic_base_retrieval_gpt-oss_120b_ranked_results_t0.5.json"
)

with open(data_file, "r") as f:
    data = json.load(f)

with open(base_data_file, "r") as f:
    base_data = json.load(f)

with open(agentic_file, "r") as f:
    agentic_data = json.load(f)

with open(agentic_with_base_only_file, "r") as f:
    agentic_base_only_data = json.load(f)

# --- Process agentic predictions ---
for idx, pred in enumerate(agentic_data):
    ranked_files = pred.get("ranked_scores", {})
    ranked_files = [
        file for file, _ in sorted(ranked_files.items(), key=lambda x: -x[1])
    ][:10]
    agentic_data[idx]["retrieved_files_agentic"] = ranked_files
    patch = pred.get("patch_file", "")
    agentic_data[idx]["is_patch_in_top10_agentic"] = patch in ranked_files[:10]

for idx, pred in enumerate(agentic_base_only_data):
    ranked_files = pred.get("ranked_scores", {})
    ranked_files = [
        file for file, _ in sorted(ranked_files.items(), key=lambda x: -x[1])
    ][:10]
    agentic_base_only_data[idx]["retrieved_files_agentic"] = ranked_files
    patch = pred.get("patch_file", "")
    agentic_base_only_data[idx]["is_patch_in_top10_agentic"] = (
        patch in ranked_files[:10]
    )

# --- Plot for multiple RANKs ---
RANKS = [1, 3, 10]
fig, axes = plt.subplots(1, len(RANKS), figsize=(6 * len(RANKS), 6))

# Use the same hit_sets keys for legend
legend_labels = [
    "Actual Bug Report",
    "Query Transformation (T0)",
    "Query Transformation (T1)",
    # "Agentic RAG (Base)",
    # "Agentic RAG (T0 + T1)",
]

for i, RANK in enumerate(RANKS):
    # Compute hit sets
    A_base = {
        d["swe_data_index"]
        for d in base_data
        if d["patch_file"] in d["retrieved_files"][:RANK]
    }
    B_t0 = {
        d["swe_data_index"]
        for d in data
        if d["patch_file"] in d["retrieved_files_t0"][:RANK]
    }
    C_t1 = {
        d["swe_data_index"]
        for d in data
        if d["patch_file"] in d["retrieved_files_t1"][:RANK]
    }
    D_agentic = {
        d["swe_data_index"]
        for d in agentic_base_only_data
        if d["patch_file"] in d["retrieved_files_agentic"][:RANK]
    }
    E_agentic = {
        d["swe_data_index"]
        for d in agentic_data
        if d["patch_file"] in d["retrieved_files_agentic"][:RANK]
    }

    hit_sets = {
        "Actual Bug Report": A_base,
        "Query Transformation (T0)": B_t0,
        "Query Transformation (T1)": C_t1,
        # "Agentic Ranking (Base)": D_agentic,
        # "Agentic Ranking (T0 + T1)": E_agentic,
    }

    ax = axes[i]
    venn(hit_sets, ax=ax)

    # Remove the legend from this subplot
    legend = ax.get_legend()
    if legend:
        legend.remove()

    # Add subfigure caption below each subplot
    ax.text(
        0.5,
        1.0,
        f"Top-{RANK}",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=2,
    )

    for text in ax.texts:
        text.set_fontsize(14)

# Add one common legend below all subplots
fig.legend(legend_labels, loc="lower center", ncol=5, fontsize=14, frameon=False)

plt.tight_layout()
# fig.subplots_adjust(bottom=0.2)  # leave space for legend
fig.savefig("figures/venn_diagrams_ranks_test.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
