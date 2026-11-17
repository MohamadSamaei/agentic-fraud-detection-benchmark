import json
import yaml
import pandas as pd
import numpy as np
import shap
import os
import time
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

from src.data_processor import create_advanced_contextual_features, add_global_features, prepare_data_for_xgboost
from src.agent_core import create_workflow, extract_agent_reviewer_payload
from src.evaluator import calculate_transaction_cost
from pathlib import Path


load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
#rebuilding the same sample_df with random_state = 42
df_train_raw = pd.read_csv(config["paths"]["train_data"])
df_test_raw = pd.read_csv(config["paths"]["test_data"])
df_train_raw["is_train_set"], df_test_raw["is_train_set"] = 1, 0

df_all = create_advanced_contextual_features(pd.concat([df_train_raw, df_test_raw]))
train_df = add_global_features(df_all[df_all["is_train_set"] == 1].drop(columns=["is_train_set"]).copy())
test_df = add_global_features(df_all[df_all["is_train_set"] == 0].drop(columns=["is_train_set"]).copy())
x_train, y_train, _, _, drop_columns = prepare_data_for_xgboost(train_df, test_df)

x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
pos_weight = (len(y_tr) - sum(y_tr)) / sum(y_tr)
xgb_model = XGBClassifier(scale_pos_weight=pos_weight, enable_categorical=True, random_state=42, n_estimators=config["model"]["xgb_estimators"], max_depth=config["model"]["xgb_max_depth"])
xgb_model.fit(x_tr, y_tr)

val_probs = xgb_model.predict_proba(x_val)[:, 1]
best_threshold = min(np.arange(0.01, 1, 0.01), key=lambda t: (config["business"]["chargeback_fee"] * confusion_matrix(y_val, (val_probs >= t).astype(int)).ravel()[2]) + (config["business"]["churn_penalty"] * confusion_matrix(y_val, (val_probs >= t).astype(int)).ravel()[1]))

fraud_samples = test_df[test_df["is_fraud"] == 1].copy()
all_legits = test_df[test_df["is_fraud"] == 0].copy()
temp_legits = all_legits.copy()
for df in [temp_legits]:
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_dayofweek"] = df["trans_date_trans_time"].dt.day_of_week
    df["trans_week"] = df["trans_date_trans_time"].dt.month
for col in ["gender", "category", "state", "merchant", "job"]:
    temp_legits[col] = temp_legits[col].astype("category")

all_legits["fraud_prob"] = xgb_model.predict_proba(temp_legits.drop(columns=drop_columns))[:, 1]
hard_legits = all_legits.sort_values(by="fraud_prob", ascending=False).head(config["benchmark"]["hard_negatives_count"]).drop(columns=["fraud_prob"])
sample_df = pd.concat([fraud_samples, hard_legits]).sample(frac=1, random_state=42).reset_index(drop=True)
sample_df["trans_hour"] = sample_df["trans_date_trans_time"].dt.hour
sample_df["trans_dayofweek"] = sample_df["trans_date_trans_time"].dt.day_of_week
sample_df["trans_week"] = sample_df["trans_date_trans_time"].dt.month
for col in ["gender", "category", "state", "merchant", "job"]:
    sample_df[col] = sample_df[col].astype("category")

safe_model_name = config["model"]["name"].replace("/", "-").replace(":", "-")

# Match exact checkpoint filename produced by evaluator.py
common_path = Path("benchmarks_errors_results")
results_path = common_path/f"results_{safe_model_name}.json"
skipped_path = common_path/f"skipped_{safe_model_name}.json"
errors_path = common_path/f"errors_{safe_model_name}.json"
benchmark_path = common_path/f"benchmark_{safe_model_name}.json"

with open(results_path, "r", encoding="utf-8") as f:
    res = json.load(f)

# Read directly from skipped_{model}.json instead of manual list
with open(skipped_path, "r", encoding="utf-8") as f:
    skipped_records = json.load(f)

misclassified_cases = []
if os.path.exists(errors_path):
    with open(errors_path, "r", encoding="utf-8") as f:
        misclassified_cases = json.load(f)

if not skipped_records:
    print("No skipped transactions found. Nothing to backfill.")
    raise SystemExit(0)

app = create_workflow(config)
explainer = shap.TreeExplainer(xgb_model)
still_skipped = []
backfill_start_time = time.time()

for item in skipped_records:
    idx = int(item["idx"])
    expected_tx_id = str(item["transaction_id"])
    actual_tx_id = str(sample_df.iloc[idx]["trans_num"])

    # [Safety Check] Verify row alignment before invoking LLM
    if expected_tx_id != actual_tx_id:
        raise ValueError(f"Row mismatch at idx {idx}: expected {expected_tx_id}, got {actual_tx_id}")

    true_label = int(sample_df.iloc[idx]["is_fraud"])
    row_features = sample_df.iloc[[idx]].drop(columns=drop_columns)
    ml_prob = float(xgb_model.predict_proba(row_features)[:, 1][0])
    ml_is_fraud = ml_prob >= best_threshold
    shap_vals = explainer.shap_values(row_features)[0]
    top_idx = np.argsort(np.abs(shap_vals))[-2:]
    top_feats = row_features.columns[top_idx].tolist()
    amt = float(row_features["amt"].iloc[0])

    initial_state = {
        "payload": extract_agent_reviewer_payload(sample_df, idx),
        "ml_label": "fraud" if ml_is_fraud else "legitimate",
        "ml_confidence": ml_prob,
        "ml_shap_summary": f"Top contributing features affecting ML decision: {top_feats[1]} and {top_feats[0]}"
    }

    # Prevent transient API failures during backfill
    state = None
    last_error = None
    for attempt in range(4):
        try:
            state = app.invoke(initial_state)
            break
        except Exception as e:
            last_error = str(e)
            print(f"Retry error at idx {idx}, Attempt {attempt + 1}: {e}")
            if attempt < 3:
                time.sleep(3 * (1 + attempt))

    if state is None:
        still_skipped.append({"idx": idx, "transaction_id": actual_tx_id, "error": last_error})
        continue

    # Update Baseline counters & cost
    b_route = "block" if ml_is_fraud else "auto_approve"
    b_key = "TP" if (b_route == "block" and true_label == 1) else "FP" if b_route == "block" else "FN" if true_label == 1 else "TN"
    res["baseline_counters"][b_key] += 1
    res["baseline_total_cost"] += calculate_transaction_cost(true_label, b_route, amt, config["business"])

    # Update Hybrid counters, error log, & operational cost
    h_route = state["final_route"]
    error_type = None
    if h_route == "human_review":
        res["hybrid_counters"]["Human_Reviews"] += 1
    elif h_route == "block":
        if true_label == 1:
            res["hybrid_counters"]["TP"] += 1
        else:
            res["hybrid_counters"]["FP"] += 1
            error_type = "FP"
    elif h_route == "auto_approve":
        if true_label == 0:
            res["hybrid_counters"]["TN"] += 1
        else:
            res["hybrid_counters"]["FN"] += 1
            error_type = "FN"

    # Record any newly recovered FP/FN cases
    if error_type is not None:
        misclassified_cases.append({
            "idx": idx,
            "transaction_id": actual_tx_id,
            "error_type": error_type,
            "true_label": true_label,
            "final_route": h_route,
            "amount": amt,
            "ml_prob": round(ml_prob, 4),
            "reviewer_verdict": state.get("reviewer_verdict"),
            "reviewer_confidence": state.get("reviewer_confidence"),
            "reviewer_rationale": state.get("reviewer_rationale"),
            "critic_recommendation": state.get("critic_recommendation"),
            "critic_counter_evidence": state.get("critic_counter_evidence"),
            "critic_supporting_evidence": state.get("critic_supporting_evidence")
        })

    calls = 2 if state.get("critic_recommendation") is not None else 1
    res["hybrid_counters"]["LLM_Calls"] += calls
    res["hybrid_counters"]["Input_Tokens"] += (state.get("reviewer_input_tokens") or 0) + (state.get("critic_input_tokens") or 0)
    res["hybrid_counters"]["Output_Tokens"] += (state.get("reviewer_output_tokens") or 0) + (state.get("critic_output_tokens") or 0)
    res["hybrid_counters"]["Total_Latency_Sec"] += (state.get("reviewer_latency") or 0.0) + (state.get("critic_latency") or 0.0)
    res["hybrid_total_cost"] += calculate_transaction_cost(true_label, h_route, amt, config["business"])

    print(f"Recovered idx {idx} -> {h_route}")
    time.sleep(config["benchmark"]["api_delay_seconds"])

# Persist updated checkpoint, error log, and remaining skipped cases
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=4)

with open(errors_path, "w", encoding="utf-8") as f:
    json.dump(sorted(misclassified_cases, key=lambda x: x["idx"]), f, indent=4)

with open(skipped_path, "w", encoding="utf-8") as f:
    json.dump(still_skipped, f, indent=4)

# Keep final KPIs and ROI metrics fully synchronized
b_cnt = res["baseline_counters"]
h_cnt = res["hybrid_counters"]
total_evaluated = b_cnt["TP"] + b_cnt["FP"] + b_cnt["TN"] + b_cnt["FN"]
total_actual_frauds = b_cnt["TP"] + b_cnt["FN"]

api_input_cost = (h_cnt["Input_Tokens"] / 1_000_000) * config["benchmark"]["cost_per_1m_input"]
api_output_cost = (h_cnt["Output_Tokens"] / 1_000_000) * config["benchmark"]["cost_per_1m_output"]
total_api_cost = api_input_cost + api_output_cost
hybrid_total_with_api = res["hybrid_total_cost"] + total_api_cost

gross_savings = res["baseline_total_cost"] - res["hybrid_total_cost"]
net_savings = res["baseline_total_cost"] - hybrid_total_with_api
roi_multiplier = (gross_savings / total_api_cost) if total_api_cost > 0 else 0.0

fp_reduction_pct = ((b_cnt["FP"] - h_cnt["FP"]) / b_cnt["FP"] * 100) if b_cnt["FP"] > 0 else 0.0
baseline_fn_rate = (b_cnt["FN"] / total_actual_frauds * 100) if total_actual_frauds > 0 else 0.0
hybrid_fn_rate = (h_cnt["FN"] / total_actual_frauds * 100) if total_actual_frauds > 0 else 0.0
human_review_rate = (h_cnt["Human_Reviews"] / total_evaluated * 100) if total_evaluated > 0 else 0.0
frauds_in_human_queue = total_actual_frauds - (h_cnt["TP"] + h_cnt["FN"])
queue_fraud_rate = (frauds_in_human_queue / h_cnt["Human_Reviews"] * 100) if h_cnt["Human_Reviews"] > 0 else 0.0

prev_exec_mins = 0.0
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        prev_bench = json.load(f)
        prev_exec_mins = prev_bench.get("engineering_metrics", {}).get("total_execution_minutes", 0.0)

total_exec_mins = prev_exec_mins + ((time.time() - backfill_start_time) / 60.0)
safe_minutes = total_exec_mins if total_exec_mins > 0 else 1e-6
avg_latency = (h_cnt["Total_Latency_Sec"] / h_cnt["LLM_Calls"]) if h_cnt["LLM_Calls"] > 0 else 0.0

benchmark_summary = {
    "metadata": {
        "model_name": config["model"]["name"],
        "provider": config["model"]["provider"],
        "ml_threshold": round(float(best_threshold), 4),
        "economic_threshold": config["model"]["economic_threshold"],
        "total_dataset_samples": len(sample_df),
        "evaluated_samples": total_evaluated,
        "skipped_samples": len(still_skipped)
    },
    "confusion_matrices": {
        "baseline": b_cnt,
        "hybrid": h_cnt
    },
    "operational_kpis": {
        "baseline_auto_fn_rate_pct": round(baseline_fn_rate, 2),
        "hybrid_auto_fn_rate_pct": round(hybrid_fn_rate, 2),
        "fp_reduction_pct": round(fp_reduction_pct, 2),
        "human_review_rate_pct": round(human_review_rate, 2),
        "queue_fraud_rate_pct": round(queue_fraud_rate, 2)
    },
    "engineering_metrics": {
        "throughput_tpm": round((h_cnt["Input_Tokens"] + h_cnt["Output_Tokens"]) / safe_minutes, 1),
        "throughput_rpm": round(h_cnt["LLM_Calls"] / safe_minutes, 2),
        "avg_pure_llm_latency_sec": round(avg_latency, 3),
        "total_execution_minutes": round(total_exec_mins, 2),
        "api_delay_seconds": config["benchmark"]["api_delay_seconds"]
    },
    "financial_roi_metrics": {
        "baseline_total_cost_usd": round(res["baseline_total_cost"], 4),
        "hybrid_operational_cost_usd": round(res["hybrid_total_cost"], 4),
        "total_api_cost_usd": round(total_api_cost, 4),
        "hybrid_total_cost_usd": round(hybrid_total_with_api, 4),
        "net_savings_usd": round(net_savings, 4),
        "roi_multiplier": round(roi_multiplier, 1)
    }
}

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_summary, f, indent=4)

print(f"Backfill completed! Recovered: {len(skipped_records) - len(still_skipped)}, Remaining skipped: {len(still_skipped)}")