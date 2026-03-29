import json

import matplotlib.pyplot as plt
from venn import venn


agentless_data_file = "repair_results/run_2/agentless/agentless_majority_voting_and_regression_and_reproduction_evaluation.json"
agentless_data_file_2 = "repair_results/run_1/agentless/agentless_majority_voting_and_regression_and_reproduction_evaluation.json"

rag_data_file_4 = "repair_results/run_1/ragent/ragent_majority_voting_and_regression_and_reproduction_evaluation.json"
rag_data_file_3 = "repair_results/run_2/ragent/ragent_majority_voting_and_regression_and_reproduction_evaluation.json"

with open(agentless_data_file, "r") as f:
    agentless_data_1 = json.load(f)

with open(agentless_data_file_2, "r") as f:
    agentless_data_2 = json.load(f)

with open(rag_data_file_3, "r") as f:
    ragfix_data_1 = json.load(f)
with open(rag_data_file_4, "r") as f:
    ragfix_data_2 = json.load(f)

# --- Prepare sets ---
A = set(agentless_data_1["resolved_ids"])
B = set(agentless_data_2["resolved_ids"])
C = set(ragfix_data_1["resolved_ids"])
D = set(ragfix_data_2["resolved_ids"])

# exit()

# --- Create figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- First Venn plot ---
venn_data_1 = {
    "Agentless (Run 1)": A,
    "Agentless (Run 2)": B,
}
venn(venn_data_1, ax=ax1)
ax1.legend(
    ["Agentless (Run 1)", "Agentless (Run 2)"],
    loc="lower center",
    ncol=2,
    fontsize=12,
    frameon=False,
)
# ax1.set_title("Overlap of Resolved Bugs", fontsize=14)

# --- Second Venn plot ---
venn_data_2 = {"Set C": C, "Set D": D}
venn(venn_data_2, ax=ax2)
ax2.legend(
    ["Agentic RAG (Run 1)", "Agentic RAG (Run 2)"],
    loc="lower center",
    ncol=2,
    fontsize=12,
    frameon=False,
)
# ax2.set_title("Second Venn Diagram", fontsize=14)

# Save plot
plt.tight_layout()
plt.savefig("repair_overlaps.pdf", dpi=300, bbox_inches="tight")
plt.close()
