import pandas as pd
import time
import json
import os
import httpx
from dotenv import load_dotenv
from typing import Dict, Any, List, Literal, TypedDict, Optional, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

# different provides
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # pyright: ignore[reportMissingImports]
from langchain_anthropic import ChatAnthropic # pyright: ignore[reportMissingImports]
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter


# Safe helper for both reasoning and non-reasoning models
def _extract_reasoning_tokens(raw_msg) -> int:
    """Safely extract reasoning token count from LangChain or OpenRouter message metadata.

    Args:
        raw_msg (Optional[AIMessage]): Raw LLM response message.

    Returns:
        int: Consumed reasoning tokens, or `0` if unavailable.
    """
    if raw_msg is None:
        return 0
    # 1. Check standardized LangChain usage_metadata
    usage = getattr(raw_msg, "usage_metadata", None) or {}
    out_details = usage.get("output_token_details") or {}
    if out_details.get("reasoning") is not None:
        return int(out_details["reasoning"])

    # 2. Fallback to raw OpenRouter/OpenAI response_metadata
    resp_meta = getattr(raw_msg, "response_metadata", None) or {}
    token_usage = resp_meta.get("token_usage") or {}
    comp_details = token_usage.get("completion_tokens_details") or {}
    return int(comp_details.get("reasoning_tokens") or 0)

class TransactionContext(BaseModel):
    transaction_id: str
    card_id: str
    age: int
    gender: str
    job: str
    city_population: int
    timestamp:str
    amount: float
    merchant: str
    category: str
    location: str
    card_median_spend: float
    spend_ratio: float
    amount_z_score: float
    velocity_24h: float
    amount_24h: float
    merchant_familiarity: int
    category_familiarity: int
    dis_to_home_km: float
    distance_from_prev_trans_km: float
    travel_speed_kmh: float
    is_night_time: int
    time_since_last_trans_hours: Union[float, str]


def extract_agent_reviewer_payload(df: pd.DataFrame, row_index: int) -> Dict[str, Any]:
    """Extract, sanitize, and serialize a transaction row into a `TransactionContext` dictionary.

    Args:
        df (pd.DataFrame): Feature-engineered transaction DataFrame.
        row_index (int): Zero-based positional index of the target row.

    Returns:
        Dict[str, Any]: Serialized `TransactionContext` payload for the Reviewer agent.
    """
    
    row = df.iloc[row_index]

    payload = TransactionContext(
        transaction_id=str(row["trans_num"]),
        card_id = str(row["cc_num"]),
        age = int(row["age"]),
        gender = str(row["gender"]),
        job = str(row["job"]),
        city_population = int(row["city_pop"]),
        timestamp = str(row["trans_date_trans_time"]),
        amount=float(row['amt']),
        merchant=str(row['merchant']).removeprefix("fraud_"),
        category=str(row['category']),
        location=f"{row['city']}, {row['state']}",
        card_median_spend=round(float(row['card_median_spend']), 2),
        spend_ratio=round(float(row['spend_ratio']), 2),
        amount_z_score=round(float(row['amount_z_score']), 2),
        velocity_24h=int(row['velocity_24h']),
        amount_24h=round(float(row['amount_24h']), 2),
        merchant_familiarity=int(row['merchant_familiarity']),
        category_familiarity=int(row['category_familiarity']),
        dis_to_home_km=round(float(row['dis_to_home']), 2),
        distance_from_prev_trans_km = round(float(row["distance_from_prev_trans"]), 2),
        travel_speed_kmh=round(float(row['travel_speed']), 2),
        is_night_time=int(row['is_night_time']),
        time_since_last_trans_hours = (
                                        "No prior transaction history"
                                        if float(row["time_since_last_trans"]) >= 9999
                                        else round(float(row["time_since_last_trans"]), 2))
    )
    return payload.model_dump()

class ReviewerOutput(BaseModel):
    evidence_list: List[str] = Field(description="List of factual evidence. Rule: EACH item MUST reference a concrete field and value (e.g., 'amount is 8.2x the card_median_spend')")
    rationale: str = Field(description="A concise explanation connecting the evidence to the final verdict")
    verdict: Literal["fraud_suspected", "legitimate", "uncertain"] = Field(description="The independent judgment on the transaction")
    confidence: Literal["high", "medium", "low"] = Field(description="The confidence level in the verdict")
    uncertainty_flag: bool = Field(description="Set to true if there is missing or conflicting information preventing a clear verdict")
    


class CriticOutput(BaseModel):
    recommendation: Literal["support_ml", "support_reviewer", "insufficient_evidence"] = Field(description = "The recommendation on which system's judgement is better supported by the data.")
    counter_evidence: List[str] = Field(description="Evidence that directly contradicts the judgment of the side you are NOT supporting.")
    supporting_evidence: List[str] = Field(description="Evidence that strengthens the judgment of the side you ARE supporting.")
    identified_risks: List[str] = Field(description="Potential risks or red flags that either the ML model or the Reviewer might have overlooked.")
    missing_information_flags: List[str] = Field(description="List of any missing context or data points that prevent a 100% certain conclusion.")


REVIEWER_SYSTEM_PROMPT = """
You are an expert, independent financial fraud investigator.
Your task is to analyze a credit card transaction and its historical context to determine if it is legitimate or suspected fraud.

You will receive a JSON payload containing transaction details and historical context features (e.g., past spending velocity, distances, familiarities).

CRITICAL RULES:
1. INDEPENDENT JUDGMENT: You must evaluate the transaction entirely on the provided contextual evidence.
2. EVIDENCE RULE: Every claim in your 'evidence_list' MUST reference a concrete field and exact numeric or categorical value from the input.
    - BAD: "The transaction amount is unusually high."
    - GOOD: "The amount of $850.00 is 3.5x higher than the card_median_spend of $240.00."
    - GOOD: "The distance_from_prev_trans_km is 1200 km while the time_since_last_trans_hours is only 2 hours, resulting in an impossible travel_speed_kmh."
3. CARD TESTING DETECTION:
   - Card testing ONLY applies to small amounts (under $25). Never label high-value transactions (e.g., > $100) as card testing.
   - Do NOT dismiss small amounts (under $25) as harmless merely because category_familiarity is high if they coincide with rapid transaction bursts (high velocity_24h), unfamiliar merchants, or an impossible trip (as defined in Rule 7).
4. FIRST TRANSACTION HANDLING: If time_since_last_trans_hours is "No prior transaction history", treat it as the card's initial recorded transaction rather than an anomalous time gap.
5. UNCERTAINTY: If the data points in conflicting directions and you cannot confidently decide, set the verdict to "uncertain" and uncertainty_flag to true.
6. DO NOT assume any external information. Use ONLY the provided payload.
7. PHYSICAL VELOCITY REGIMES:
   - Velocity under 130 km/h is completely normal and must never be treated as suspicious on its own (including slow walking speeds of 1-10 km/h).
   - Velocity between 130 km/h and 900 km/h is normal for rail or air travel when the distance and elapsed time are realistic (e.g., intercity trips over at least 1 hour for flights).
   - An impossible trip occurs ONLY when velocity exceeds 900 km/h across a significant distance (> 50 km), or when transactions show a back-and-forth (ping-pong) pattern between distant cities within a short period.

Analyze the input, extract factual evidence, formulate your rationale, and output strictly conforming to the requested JSON schema.
"""

CRITIC_SYSTEM_PROMPT = """
You are an expert Adversarial Fraud Critic.
Your job is NOT to make the final decision, but to resolve a disagreement between a Machine Learning model and an AI Reviewer regarding a credit card transaction.

You will receive:
1. The Transaction Context (amounts, history, locations, etc.).
2. The ML model's Output (probability, label, and SHAP features).
3. The AI Reviewer's output (verdict, confidence and rationale).

CRITICAL RULES:
1. CHALLENGE EVERYTHING & AVOID SYMMETRIC ML ANCHORING:
    - Do not blindly trust the ML model or the Reviewer in EITHER direction.
    - Low-ML Anchoring Guard: Do NOT default to "support_ml" solely because the ML model's probability is near zero when the Reviewer has identified concrete behavioral anomalies (such as spend_ratio > 2.5, impossible travel, or high amount_24h).
    - High-ML Anchoring Guard: Do NOT default to "support_ml" solely because the ML probability is high when raw transaction features (normal velocity, familiar merchant/category, typical spend_ratio) support the Reviewer's "legitimate" verdict. If signals genuinely conflict, recommend "insufficient_evidence".
2. EVIDENCE-BASED CRITIQUE: If you support the ML model, provide 'counter_evidence' against the Reviewer's rationale. If you support the Reviewer, point out why the ML model's statistical anomaly might be a false positive (e.g., a legitimate travel expense).
3. CONCRETE REFERENCES: Just like the reviewer, all evidence you cite MUST reference exact numerical or categorical values from the input.
4. NO FINAL VERDICTS: You cannot approve or decline the transaction. You may only output one of three recommendations: "support_ml", "support_reviewer", or "insufficient_evidence".
5. PHYSICAL VELOCITY REGIMES:
   - Velocity under 130 km/h is completely normal and must never be treated as suspicious on its own (including slow walking speeds of 1-10 km/h).
   - Velocity between 130 km/h and 900 km/h is normal for rail or air travel when the distance and elapsed time are realistic (e.g., intercity trips over at least 1 hour for flights).
   - An impossible trip occurs ONLY when velocity exceeds 900 km/h across a significant distance (> 50 km), or when transactions show a back-and-forth (ping-pong) pattern between distant cities within a short period.

Analyze the conflicting judgments, weigh the factual evidence, and output strictly conforming to the requested JSON schema.
"""

class TransactionState(TypedDict):
    payload: dict
    ml_label: str
    ml_confidence: float
    ml_shap_summary: str

    reviewer_rationale: Optional[str]
    reviewer_verdict: Optional[str]
    reviewer_confidence: Optional[str]
    reviewer_uncertainty_flag: Optional[bool]
    reviewer_evidence_list: Optional[List[str]]
    reviewer_input_tokens: Optional[int]
    reviewer_output_tokens: Optional[int]
    reviewer_reasoning_tokens: Optional[int]
    reviewer_latency: Optional[float]

    critic_recommendation: Optional[str]
    critic_counter_evidence: Optional[List[str]]
    critic_supporting_evidence: Optional[List[str]]
    critic_identified_risks: Optional[List[str]]
    critic_missing_information_flags: Optional[List[str]]
    critic_input_tokens: Optional[int]
    critic_output_tokens: Optional[int]
    critic_reasoning_tokens: Optional[int]
    critic_latency: Optional[float]

    final_route: Optional[str]

    reviewer_token: int
    critic_tokens: int    

load_dotenv()

def get_llm(config_dict: dict):
    """Initialize a configured LangChain chat model (`ChatOpenAI` for OpenRouter or `ChatGoogleGenerativeAI`).

    Args:
        config_dict (dict): Configuration dictionary with provider, model name, temperature, and token limits.

    Returns:
        Union[ChatOpenAI, ChatGoogleGenerativeAI]: Configured LLM client instance.

    Raises:
        ValueError: If required API keys are missing or the provider is unsupported.
    """
    provider = config_dict["model"]["provider"].lower()
    model_name = config_dict["model"]["name"]
    temp = config_dict["model"].get("temperature", 0)
    max_tokens = config_dict["model"].get("max_tokens", 4000)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing in .env")

        timeout = httpx.Timeout(connect=10.0, read=25.0, write=10.0, pool=10.0)
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)

        sync_client = httpx.Client(
            timeout=timeout,
            limits=limits,
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        )
        async_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
        )

        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            max_tokens=max_tokens,
            max_retries=2,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            http_client=sync_client,
            http_async_client=async_client,
            extra_body={
                            "provider": {
                                "allow_fallbacks": False,
                            },
                            "reasoning_effort": "none",
                        }
        )

    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is missing in .env")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            max_output_tokens=max_tokens,
            max_retries=2,
            google_api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def create_workflow(config_dict: dict):
    """Build and compile the LangGraph state machine for hybrid fraud triage.

    Args:
        config_dict (dict): Configuration dictionary with model settings and decision thresholds.

    Returns:
        CompiledStateGraph: Compiled executable LangGraph workflow.
    """
    llm = get_llm(config_dict)

    reviewer_agent = llm.with_structured_output(ReviewerOutput, include_raw=True)
    critic_agent = llm.with_structured_output(CriticOutput, include_raw=True)
    model_cfg = config_dict["model"]
    economic_threshold = config_dict["model"]["economic_threshold"]
    ml_threshold = float(model_cfg.get("ml_threshold", 0.5))
    high_risk_ml_threshold = float(model_cfg.get("high_risk_ml_threshold", .65))
    ultra_low_ml_threshold = float(model_cfg.get("ultra_low_ml_threshold", .01))
    critical_amount_threshold = float(model_cfg.get("critical_amount_threshold", 500))


    def run_reviewer(state: TransactionState):
        """Run the primary Reviewer LLM agent and record structured verdict, tokens, and latency.

    Args:
        state (TransactionState): Current graph state containing the transaction `"payload"`.

    Returns:
        dict: State updates with Reviewer verdict, confidence, rationale, token counts, and latency.
    """
        messages = [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(state["payload"], indent=2))
        ]
        start_time = time.time()
        response = reviewer_agent.invoke(messages)
        latency = time.time() - start_time
        
        parsed = response["parsed"] if isinstance(response, dict) else response
        raw_message = response["raw"]
        usage = getattr(raw_message, "usage_metadata", None) or {}
        if not usage and hasattr(raw_message, "response_metadata"):
            usage = raw_message.response_metadata.get("token_usage", {})
        
        return {
            "reviewer_verdict": parsed.verdict,
            "reviewer_confidence": parsed.confidence,
            "reviewer_uncertainty_flag": parsed.uncertainty_flag,
            "reviewer_evidence_list": parsed.evidence_list,
            "reviewer_rationale": parsed.rationale,
            "reviewer_input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
            "reviewer_output_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
            "reviewer_reasoning_tokens": _extract_reasoning_tokens(raw_message),
            "reviewer_latency": latency
        }

    def should_run_critic(state: TransactionState) -> Literal["critic", "router"]:
        """Route contested, uncertain, or high-stakes transactions to the Critic agent, or skip to routing.

        Args:
            state (TransactionState): Current graph state with ML scores and Reviewer outputs.

        Returns:
            Literal["critic", "router"]: Next target node in the workflow.
        """
        ml_prob = float(state.get("ml_confidence", 0))
        ml_is_fraud = (state.get("ml_label") == "fraud") or (ml_prob >= ml_threshold)
        rev_verdict = (state.get("reviewer_verdict") or "uncertain").strip().lower()
        rev_conf = (state.get("reviewer_confidence") or "medium").strip().lower()
        rev_is_fraud = rev_verdict == "fraud_suspected"
        rev_uncertain = (rev_verdict == "uncertain") or bool(state.get("reviewer_uncertainty_flag"))
        amount = state["payload"]["amount"]

        if (ml_prob < ultra_low_ml_threshold and amount < critical_amount_threshold
            and rev_verdict == "legitimate"
            and not rev_uncertain):
            return "router"

        if amount < economic_threshold:
            return "router"
        
        disagreement = ml_is_fraud != rev_is_fraud
        if disagreement or rev_uncertain or rev_conf != "high":
            return "critic"
        
        return "router"

    def run_critic(state: TransactionState):
        """Run the secondary Critic LLM agent to audit ML and Reviewer disagreements.

        Args:
            state (TransactionState): Current graph state with transaction context, ML/SHAP output, and Reviewer verdict.

        Returns:
            dict: State updates with Critic recommendation, evidence flags, token counts, and latency.
        """
        critic_input = {
            "transaction_context": state["payload"],
            "ml_output": {
                "label": state["ml_label"],
                "confidence": state["ml_confidence"],
                "shap_summary": state["ml_shap_summary"]
            },
            "reviewer_output": {
                "verdict": state["reviewer_verdict"],
                "confidence": state["reviewer_confidence"],
                "uncertainty_flag": state["reviewer_uncertainty_flag"],
                "evidence_list": state["reviewer_evidence_list"],
                "rationale": state["reviewer_rationale"]
            }
        }
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(critic_input, indent=2))
        ]
        start_time = time.time()
        response = critic_agent.invoke(messages)
        latency = time.time() - start_time
        
        parsed = response["parsed"] if isinstance(response, dict) else response
        raw_message = response["raw"]
        usage = getattr(raw_message, "usage_metadata", None) or {}
        if not usage and hasattr(raw_message, "response_metadata"):
            usage = raw_message.response_metadata.get("token_usage", {})

        return {
            "critic_recommendation": parsed.recommendation,
            "critic_counter_evidence": parsed.counter_evidence,
            "critic_supporting_evidence": parsed.supporting_evidence,
            "critic_identified_risks": parsed.identified_risks,
            "critic_missing_information_flags": parsed.missing_information_flags,
            "critic_input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
            "critic_output_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
            "critic_reasoning_tokens": _extract_reasoning_tokens(raw_message), 
            "critic_latency": latency
        }

    def final_routing(state: TransactionState):
        """Apply deterministic policy rules to assign the final route (`auto_approve`, `block`, or `human_review`).

        Args:
            state (TransactionState): Completed graph state with ML scores, payload signals, and agent verdicts.

        Returns:
            dict: State update containing `{"final_route": route}`.
        """
        ml_prob = float(state.get("ml_confidence", 0))
        ml_is_fraud = (state.get("ml_label") == "fraud") or (ml_prob >= ml_threshold)
        
        rev_verdict = (state.get("reviewer_verdict") or "uncertain").strip().lower()
        rev_conf = (state.get("reviewer_confidence") or "medium").strip().lower()
        rev_is_fraud = rev_verdict == "fraud_suspected"
        rev_uncertain = (rev_verdict == "uncertain") or bool(state.get("reviewer_uncertainty_flag"))
        
        payload = state["payload"]
        amount = float(payload["amount"])
        critic_rec = (
            state.get("critic_recommendation").strip().lower()
            if state.get("critic_recommendation") else None
        )

        impossible_trip = (
            payload.get("travel_speed_kmh", 0) > 900 and
            payload.get("distance_from_prev_trans_km", 0) > 50
        )
        has_high_risk_signal = (
            payload.get("spend_ratio", 0) > 2.5
            or impossible_trip
            or payload.get("amount_24h", 0) > 1000
        )

        # 1. Fast-path for completely low-risk transactions
        if (
            ml_prob < ultra_low_ml_threshold
            and amount < critical_amount_threshold
            and rev_verdict == "legitimate"
            and not rev_uncertain
        ):
            return {"final_route": "auto_approve"}

        # 2. small transactions
        if amount < economic_threshold:
            if rev_is_fraud:
                route = "block"
            elif rev_verdict == "legitimate" and rev_conf == "high":
                # Prioritize high-confidence Reviewer clearance over low ML threshold to prevent false positives
                route = "auto_approve"
            elif rev_verdict == "legitimate" and rev_conf == "medium":
                route = "block" if ml_prob >= 0.95 else "auto_approve"
            elif ml_prob >= high_risk_ml_threshold:
                route = "block"
            else:
                route = "block" if ml_is_fraud else "auto_approve"
            return {"final_route": route}

        # 3. big transactions
        if rev_uncertain:
            # Auto-approve uncertain high-value cases only if Critic supports ML, ML risk is ultra-low, and no risk flags exist
            if (
                critic_rec == "support_ml"
                and not ml_is_fraud
                and ml_prob < ultra_low_ml_threshold
                and not has_high_risk_signal
            ):
                route = "auto_approve"
            elif ml_is_fraud and ml_prob >= 0.95 and has_high_risk_signal and critic_rec == "support_ml":
                route = "block"
            else:
                route = "human_review"

        elif ml_is_fraud and not rev_is_fraud:
            # Override low-probability ML flags (threshold=0.02) when Reviewer is highly confident and no risk signals trigger
            if (
                rev_verdict == "legitimate"
                and rev_conf == "high"
                and ml_prob < 0.20
                and not has_high_risk_signal
            ):
                route = "auto_approve"
            else:
                route = "human_review"

        elif not ml_is_fraud and rev_is_fraud:
            if critic_rec == "support_ml" and ml_prob < 0.20 and amount < critical_amount_threshold:
                route = "auto_approve"
            else:
                route = "human_review"

        elif ml_is_fraud and rev_is_fraud:
            route = "block"

        else:
            route = "auto_approve" if rev_conf == "high" else "human_review"

        # 4. Safeguard against false-positive auto-blocks on high-value / critical-amount transactions
        if (
            route == "block"
            and amount >= critical_amount_threshold
            and (ml_prob < high_risk_ml_threshold or rev_conf != "high")
        ):
            route = "human_review"

        return {"final_route": route}

    workflow = StateGraph(TransactionState)
    workflow.add_node("reviewer", run_reviewer)
    workflow.add_node("critic", run_critic)
    workflow.add_node("router", final_routing)
    workflow.add_edge(START, "reviewer")
    workflow.add_conditional_edges("reviewer", should_run_critic)
    workflow.add_edge("critic", "router")
    workflow.add_edge("router", END)
    
    return workflow.compile()