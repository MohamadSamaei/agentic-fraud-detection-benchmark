import asyncio
import json
import time
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm
from src.agent_core import extract_agent_reviewer_payload
from pathlib import Path


def calculate_transaction_cost(true_label, predicted_route, amount, cfg):
    """Compute the USD financial cost of a routing decision using the business cost matrix.

    Args:
        true_label (int): Ground-truth label (`1` for fraud, `0` for legitimate).
        predicted_route (str): Assigned route (`"block"`, `"auto_approve"`, or `"human_review"`).
        amount (float): Transaction amount in USD.
        cfg (dict): Cost parameters (`profit_margin`, `churn_penalty`, `chargeback_fee`, `human_review_cost`).

    Returns:
        float: Incurred financial cost in USD.
    """
    if predicted_route == "block":
        if true_label == 1:
            return 0.0
        else:
            return (amount * cfg["profit_margin"]) + cfg["churn_penalty"]
    elif predicted_route == "auto_approve":
        if true_label == 0:
            return 0.0
        else:
            return amount + cfg["chargeback_fee"]
    elif predicted_route == "human_review":
        return cfg["human_review_cost"]


async def _evaluate_single_transaction(
    idx: int,
    initial_state: dict,
    true_label: int,
    transaction_amount: float,
    trans_num: str,
    ml_is_fraud: bool,
    ml_prob: float,
    app,
    semaphore: asyncio.Semaphore,
    api_delay: float,
    max_attempts: int = 4,
):
    """Asynchronously evaluate a single transaction with concurrency limits and exponential backoff.

    Args:
        idx (int): Dataset row index of the transaction.
        initial_state (dict): Initial LangGraph state payload.
        true_label (int): Ground-truth fraud label (`1` or `0`).
        transaction_amount (float): Transaction amount in USD.
        trans_num (str): Unique transaction identifier.
        ml_is_fraud (bool): Baseline XGBoost binary fraud prediction.
        ml_prob (float): Baseline XGBoost fraud probability.
        app: Compiled LangGraph workflow instance.
        semaphore (asyncio.Semaphore): Concurrency limiter for parallel API calls.
        api_delay (float): Post-invocation sleep duration in seconds for rate-limit pacing.
        max_attempts (int, optional): Maximum retry attempts on failure. Defaults to 4.

    Returns:
        dict: Evaluation record containing `final_state`, `last_error`, and transaction metadata.
    """
    
    async with semaphore:
        final_state = None
        last_error = None

        for attempt in range(max_attempts):
            try:
                final_state = await app.ainvoke(initial_state)
                break
            except Exception as e:
                last_error = str(e)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 * (2 ** attempt))  # Backoff: 2s, 4s, 8s

        if api_delay > 0:
            await asyncio.sleep(api_delay)

        return {
            "idx": idx,
            "final_state": final_state,
            "last_error": last_error,
            "true_label": true_label,
            "transaction_amount": transaction_amount,
            "trans_num": trans_num,
            "ml_is_fraud": ml_is_fraud,
            "ml_prob": ml_prob,
        }


async def _evaluate_policy_costs_async(test_df, xgb_model, threshold, drop_columns, app, config):
    """Run an async batch benchmark comparing baseline XGBoost against the hybrid agent workflow.

    Mines hard negatives, computes SHAP feature attributions, evaluates transactions
    concurrently with periodic checkpointing, and saves detailed JSON reports
    (confusion matrices, token/latency telemetry, operational KPIs, and financial ROI)
    to `benchmarks_errors_results/`.

    Args:
        test_df (pd.DataFrame): Feature-engineered test dataset with `is_fraud` labels.
        xgb_model: Trained XGBoost classifier.
        threshold (float): Decision probability threshold for the baseline ML model.
        drop_columns (List[str]): Non-feature columns excluded before ML inference.
        app: Compiled LangGraph workflow instance.
        config (dict): Configuration dictionary with model, business cost, and benchmark settings.

    Returns:
        Tuple[float, float]: `(hybrid_total_cost, baseline_total_cost)` in USD.
    """
    output_dir = Path("benchmarks_errors_results")
    output_dir.mkdir(parents = True, exist_ok = True)
    
    print("Mining hard negatives...")
    fraud_samples = test_df[test_df["is_fraud"] == 1].copy()
    all_legits = test_df[test_df["is_fraud"] == 0].copy()

    temp_legits = all_legits.copy()
    temp_legits["trans_hour"] = temp_legits["trans_date_trans_time"].dt.hour
    temp_legits["trans_dayofweek"] = temp_legits["trans_date_trans_time"].dt.day_of_week
    temp_legits["trans_week"] = temp_legits["trans_date_trans_time"].dt.month

    cat_cols = ["gender", "category", "state", "merchant", "job"]
    for col in cat_cols:
        temp_legits[col] = temp_legits[col].astype("category")

    x_legits = temp_legits.drop(columns=drop_columns)
    all_legits["fraud_prob"] = xgb_model.predict_proba(x_legits)[:, 1]

    num_hard = config["benchmark"]["hard_negatives_count"]
    hard_legits_samples = (
        all_legits.sort_values(by="fraud_prob", ascending=False)
        .head(num_hard)
        .drop(columns=["fraud_prob"])
    )
    sample_df = (
        pd.concat([fraud_samples, hard_legits_samples])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    sample_df["trans_hour"] = sample_df["trans_date_trans_time"].dt.hour
    sample_df["trans_dayofweek"] = sample_df["trans_date_trans_time"].dt.day_of_week
    sample_df["trans_week"] = sample_df["trans_date_trans_time"].dt.month
    for col in cat_cols:
        sample_df[col] = sample_df[col].astype("category")


    all_features = sample_df.drop(columns=drop_columns)
    all_ml_probs = xgb_model.predict_proba(all_features)[:, 1]

    explainer = shap.TreeExplainer(xgb_model)
    all_shap_vals = explainer.shap_values(all_features)
    if isinstance(all_shap_vals, list):
        all_shap_vals = all_shap_vals[1]

    feature_names = all_features.columns

    hybrid_total_cost = 0.0
    baseline_total_cost = 0.0
    hybrid_counters = {
        "TP": 0, "FP": 0, "TN": 0, "FN": 0,
        "Human_Reviews": 0, "LLM_Calls": 0,
        "Input_Tokens": 0, "Output_Tokens": 0,
        "Reviewer_Reasoning_Tokens": 0,
        "Critic_Reasoning_Tokens": 0,    
        "Total_Latency_Sec": 0.0,
    }
    baseline_counters = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "Human_Reviews": 0, "LLM_Calls": 0}
    misclassified_cases = []
    skipped_cases = []

    max_concurrency = config["benchmark"].get("max_concurrency", 15)
    api_delay = config["benchmark"].get("api_delay_seconds", 0.0)
    semaphore = asyncio.Semaphore(max_concurrency)
    safe_model_name = config["model"]["name"].replace("/", "-").replace(":", "-")

    print(f"Starting async batch evaluation on {len(sample_df)} transactions (concurrency={max_concurrency})...")
    batch_start_time = time.time()

    tasks = []
    for idx in range(len(sample_df)):
        true_label = int(sample_df.iloc[idx]["is_fraud"])
        ml_prob = float(all_ml_probs[idx])
        ml_is_fraud = ml_prob >= threshold

        shap_vals = all_shap_vals[idx]
        top_indices = np.argsort(np.abs(shap_vals))[-2:]
        top_features = feature_names[top_indices].tolist()
        dynamic_shap_summary = (
            f"Top contributing features affecting ML decision: {top_features[1]} and {top_features[0]}"
        )
        transaction_amount = float(all_features["amt"].iloc[idx])

        initial_state = {
            "payload": extract_agent_reviewer_payload(sample_df, idx),
            "ml_label": "fraud" if ml_is_fraud else "legitimate",
            "ml_confidence": ml_prob,
            "ml_shap_summary": dynamic_shap_summary,
        }

        tasks.append(
            asyncio.create_task(
                _evaluate_single_transaction(
                    idx=idx,
                    initial_state=initial_state,
                    true_label=true_label,
                    transaction_amount=transaction_amount,
                    trans_num=str(sample_df.iloc[idx]["trans_num"]),
                    ml_is_fraud=ml_is_fraud,
                    ml_prob=ml_prob,
                    app=app,
                    semaphore=semaphore,
                    api_delay=api_delay,
                )
            )
        )

    completed_count = 0
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        res = await future
        completed_count += 1

        idx = res["idx"]
        final_state = res["final_state"]
        true_label = res["true_label"]
        transaction_amount = res["transaction_amount"]
        ml_is_fraud = res["ml_is_fraud"]
        ml_prob = res["ml_prob"]

        if final_state is None:
            print(f"\nSkipping transaction {idx}: {res['last_error']}")
            skipped_cases.append({
                "idx": idx,
                "transaction_id": res["trans_num"],
                "error": res["last_error"],
            })
            file_path = output_dir/f"skipped_{safe_model_name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sorted(skipped_cases, key=lambda x: x["idx"]), f, indent=4)
        else:
            baseline_route = "block" if ml_is_fraud else "auto_approve"
            if baseline_route == "block":
                baseline_counters["TP" if true_label == 1 else "FP"] += 1
            else:
                baseline_counters["FN" if true_label == 1 else "TN"] += 1

            baseline_total_cost += calculate_transaction_cost(
                true_label, baseline_route, transaction_amount, config["business"]
            )

            sample_requests = 1
            hybrid_counters["Input_Tokens"] += final_state.get("reviewer_input_tokens", 0)
            hybrid_counters["Output_Tokens"] += final_state.get("reviewer_output_tokens", 0)
            hybrid_counters["Reviewer_Reasoning_Tokens"] += final_state.get("reviewer_reasoning_tokens", 0)
            hybrid_counters["Total_Latency_Sec"] += final_state.get("reviewer_latency", 0.0)

            if final_state.get("critic_recommendation") is not None:
                sample_requests += 1
                hybrid_counters["Input_Tokens"] += final_state.get("critic_input_tokens", 0)
                hybrid_counters["Output_Tokens"] += final_state.get("critic_output_tokens", 0)
                hybrid_counters["Critic_Reasoning_Tokens"] += final_state.get("critic_reasoning_tokens", 0)
                hybrid_counters["Total_Latency_Sec"] += final_state.get("critic_latency", 0.0)

            hybrid_counters["LLM_Calls"] += sample_requests
            decision = final_state["final_route"]
            error_type = None

            if decision == "human_review":
                hybrid_counters["Human_Reviews"] += 1
            elif decision == "block":
                if true_label == 1:
                    hybrid_counters["TP"] += 1
                else:
                    hybrid_counters["FP"] += 1
                    error_type = "FP"
            elif decision == "auto_approve":
                if true_label == 0:
                    hybrid_counters["TN"] += 1
                else:
                    hybrid_counters["FN"] += 1
                    error_type = "FN"

            if error_type is not None:
                misclassified_cases.append({
                    "idx": idx,
                    "transaction_id": res["trans_num"],
                    "error_type": error_type,
                    "true_label": true_label,
                    "final_route": decision,
                    "amount": transaction_amount,
                    "ml_prob": round(ml_prob, 4),
                    "reviewer_verdict": final_state.get("reviewer_verdict"),
                    "reviewer_confidence": final_state.get("reviewer_confidence"),
                    "reviewer_rationale": final_state.get("reviewer_rationale"),
                    "critic_recommendation": final_state.get("critic_recommendation"),
                    "critic_counter_evidence": final_state.get("critic_counter_evidence"),
                    "critic_supporting_evidence": final_state.get("critic_supporting_evidence"),
                })

            hybrid_total_cost += calculate_transaction_cost(
                true_label, decision, transaction_amount, config["business"]
            )

        # Checkpointing every 50 samples
        if completed_count % 50 == 0 or completed_count == len(sample_df):
            checkpoint_data = {
                "processed_samples": completed_count,
                "total_samples": len(sample_df),
                "baseline_counters": baseline_counters,
                "hybrid_counters": hybrid_counters,
                "baseline_total_cost": baseline_total_cost,
                "hybrid_total_cost": hybrid_total_cost,
            }
            file_path = output_dir/f"results_{safe_model_name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4)

            file_path = output_dir/f"errors_{safe_model_name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sorted(misclassified_cases, key=lambda x: x["idx"]), f, indent=4)

    batch_end_time = time.time()
    total_execution_minutes = (batch_end_time - batch_start_time) / 60.0

    api_input_cost = (hybrid_counters["Input_Tokens"] / 1_000_000) * config["benchmark"]["cost_per_1m_input"]
    api_output_cost = (hybrid_counters["Output_Tokens"] / 1_000_000) * config["benchmark"]["cost_per_1m_output"]
    total_api_cost = api_input_cost + api_output_cost
    hybrid_total_cost += total_api_cost

    print("\n=== 📊 Final Comparison ===")
    print(f"Metrics Baseline: TP={baseline_counters['TP']}, FP={baseline_counters['FP']}, TN={baseline_counters['TN']}, FN={baseline_counters['FN']}")
    print(f"Metrics Hybrid: TP={hybrid_counters['TP']}, FP={hybrid_counters['FP']}, TN={hybrid_counters['TN']}, FN={hybrid_counters['FN']}")

    print("\n=== 🚀 ENGINEERING & PERFORMANCE METRICS ===")
    safe_minutes = total_execution_minutes if total_execution_minutes > 0 else 1e-6
    tpm = (hybrid_counters["Input_Tokens"] + hybrid_counters["Output_Tokens"]) / safe_minutes
    rpm = hybrid_counters["LLM_Calls"] / safe_minutes
    avg_latency = hybrid_counters["Total_Latency_Sec"] / hybrid_counters["LLM_Calls"] if hybrid_counters["LLM_Calls"] > 0 else 0.0

    print(f"System Throughput (Tokens): {tpm:,.0f} TPM (Concurrency={max_concurrency}, Delay={api_delay}s)")
    print(f"System Throughput (Requests): {rpm:,.1f} RPM")
    print(f"Average Pure LLM Latency: {avg_latency:.2f} seconds/call")
    print(f"Total Execution Time: {total_execution_minutes:.2f} mins")
    print(f"Total API Cost: ${total_api_cost:.4f}")
    print(f"Reviewer Reasoning Tokens: {hybrid_counters['Reviewer_Reasoning_Tokens']:,}")
    print(f"Critic Reasoning Tokens: {hybrid_counters['Critic_Reasoning_Tokens']:,}")

    print("\n=== 💼 BUSINESS IMPACT & ROI METRICS ===")
    fp_reduction_pct = ((baseline_counters["FP"] - hybrid_counters["FP"]) / baseline_counters["FP"]) * 100 if baseline_counters["FP"] > 0 else 0.0
    print(f"Alert Fatigue Reduction (Drop in False Positives): {fp_reduction_pct:.1f}%")

    gross_savings = baseline_total_cost - (hybrid_total_cost - total_api_cost)
    net_savings = baseline_total_cost - hybrid_total_cost
    roi_multiplier = (gross_savings / total_api_cost) if total_api_cost > 0 else 0
    print(f"Return on AI Investment (ROI): {roi_multiplier:.1f}x")
    print(f"Net Savings by Hybrid System: ${net_savings:.2f}")

    total_evaluated = (
        baseline_counters["TP"] + baseline_counters["FP"]
        + baseline_counters["TN"] + baseline_counters["FN"]
    )
    total_actual_frauds = baseline_counters["TP"] + baseline_counters["FN"]

    baseline_fn_rate = (baseline_counters["FN"] / total_actual_frauds * 100) if total_actual_frauds > 0 else 0.0
    hybrid_fn_rate = (hybrid_counters["FN"] / total_actual_frauds * 100) if total_actual_frauds > 0 else 0.0
    human_review_rate = (hybrid_counters["Human_Reviews"] / total_evaluated * 100) if total_evaluated > 0 else 0.0

    frauds_in_human_queue = total_actual_frauds - (hybrid_counters["TP"] + hybrid_counters["FN"])
    queue_fraud_rate = (
        (frauds_in_human_queue / hybrid_counters["Human_Reviews"] * 100)
        if hybrid_counters["Human_Reviews"] > 0 else 0.0
    )

    benchmark_summary = {
        "metadata": {
            "model_name": config["model"]["name"],
            "provider": config["model"]["provider"],
            "ml_threshold": round(float(threshold), 4),
            "economic_threshold": config["model"]["economic_threshold"],
            "max_concurrency": max_concurrency,
            "total_dataset_samples": len(sample_df),
            "evaluated_samples": total_evaluated,
            "skipped_samples": len(skipped_cases),
        },
        "confusion_matrices": {
            "baseline": baseline_counters,
            "hybrid": hybrid_counters,
        },
        "operational_kpis": {
            "baseline_auto_fn_rate_pct": round(baseline_fn_rate, 2),
            "hybrid_auto_fn_rate_pct": round(hybrid_fn_rate, 2),
            "fp_reduction_pct": round(fp_reduction_pct, 2),
            "human_review_rate_pct": round(human_review_rate, 2),
            "queue_fraud_rate_pct": round(queue_fraud_rate, 2),
        },
        "engineering_metrics": {
            "throughput_tpm": round(tpm, 1),
            "throughput_rpm": round(rpm, 2),
            "avg_pure_llm_latency_sec": round(avg_latency, 3),
            "total_execution_minutes": round(total_execution_minutes, 2),
            "api_delay_seconds": api_delay,
            "reviewer_reasoning_tokens": hybrid_counters["Reviewer_Reasoning_Tokens"],
            "critic_reasoning_tokens": hybrid_counters["Critic_Reasoning_Tokens"],      
        },
        "financial_roi_metrics": {
            "baseline_total_cost_usd": round(baseline_total_cost, 4),
            "hybrid_operational_cost_usd": round(hybrid_total_cost - total_api_cost, 4),
            "total_api_cost_usd": round(total_api_cost, 4),
            "hybrid_total_cost_usd": round(hybrid_total_cost, 4),
            "net_savings_usd": round(net_savings, 4),
            "roi_multiplier": round(roi_multiplier, 1),
        },
    }

    file_path = output_dir/f"benchmark_{safe_model_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=4)

    return hybrid_total_cost, baseline_total_cost


def evaluate_policy_costs(test_df, xgb_model, threshold, drop_columns, app, config):
    """Wrapper main.py."""
    return asyncio.run(
        _evaluate_policy_costs_async(test_df, xgb_model, threshold, drop_columns, app, config)
    )