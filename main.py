"""Entry point for training the baseline XGBoost model and benchmarking the hybrid fraud detection workflow."""
import os
import yaml
import pandas as pd
import numpy as np
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

from src.data_processor import create_advanced_contextual_features, add_global_features, prepare_data_for_xgboost
from src.agent_core import create_workflow
from src.evaluator import evaluate_policy_costs

def load_config():
    """Load model, benchmark, and business cost settings from `config.yaml`.

    Returns:
        dict: Parsed YAML configuration dictionary.
    """
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # loading the settings
    load_dotenv()
    config = load_config()

    print("Loading datasets...")
    df_train_raw = pd.read_csv(config["paths"]["train_data"])
    df_test_raw = pd.read_csv(config["paths"]["test_data"])

    df_train_raw["is_train_set"] = 1
    df_test_raw["is_train_set"] = 0

    df_all = pd.concat([df_train_raw, df_test_raw])
    df_all = create_advanced_contextual_features(df_all)

    df_train = df_all[df_all["is_train_set"] == 1].drop(columns=["is_train_set"]).copy()
    df_test = df_all[df_all["is_train_set"] == 0].drop(columns=["is_train_set"]).copy()

    train_df = add_global_features(df_train)
    test_df = add_global_features(df_test)

    x_train, y_train, x_test, y_test, drop_columns = prepare_data_for_xgboost(train_df, test_df)

    # training the baseline model
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=.2, random_state=42)
    pos_weight = (len(y_tr) - sum(y_tr)) / sum(y_tr)

    print("Training XGBoost baseline...")
    xgb_model = XGBClassifier(
        scale_pos_weight=pos_weight, enable_categorical=True, 
        random_state=42, n_estimators=config["model"]["xgb_estimators"], 
        max_depth=config["model"]["xgb_max_depth"]
    )
    xgb_model.fit(x_tr, y_tr)
    val_probs = xgb_model.predict_proba(x_val)[:, 1]

    best_threshold = 0.5
    min_cost = float("inf")
    # calculating the best threshold
    c_fn = config["business"]["chargeback_fee"]
    c_fp = config["business"]["churn_penalty"]

    for threshold in np.arange(.01, 1, .01):
        preds = (val_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        current_cost = (c_fn * fn) + (c_fp * fp)
        if current_cost < min_cost:
            min_cost = current_cost
            best_threshold = threshold

    print(f"Optimal Operating Threshold: {best_threshold:.2f}")

    # running the system 
    print("Compiling LangGraph Workflow...")
    app = create_workflow(config)

    print("Running Benchmark...")
    hybrid_cost, baseline_cost = evaluate_policy_costs(test_df, xgb_model, best_threshold, drop_columns, app, config)
    
    print(f"\nNet Savings by Hybrid System: ${baseline_cost - hybrid_cost:.2f}")