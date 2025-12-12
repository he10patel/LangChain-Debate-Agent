import os

os.environ["LANGSMITH_DISABLE_RUN_COMPRESSION"] = "true"
# Ensure traces are logged to a specific project (avoids 'default' project permission issues)
os.environ["LANGCHAIN_PROJECT"] = "debate-agent-evaluation"

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from graph.build_graph import build_debate_graph
from llm.model import get_llm

client = Client()

def run_debate_agent(inputs: dict) -> dict:
    topic = inputs["topic"]
    
    initial_state = {
        "topic": topic,
        "topic_type": None,
        "history": [],
        "agent_a_last": None,
        "agent_b_last": None,
        "continue_debate": True,
        "final_response": None,
        "round_number": 1,
        "max_rounds": 3,
    }
    
    graph = build_debate_graph()
    result = graph.invoke(initial_state)
    
    return {
        "topic_type": result.get("topic_type"),
        "final_response": result.get("final_response"),
        "history": result.get("history", []),
        "num_rounds": result.get("round_number", 0)
    }

def check_classification(run, example, client=None) -> dict:
    expected_type = example.outputs.get("expected_type")
    actual_type = run.outputs.get("topic_type")

    score = 1.0 if actual_type == expected_type else 0.0
    
    return {
        "key": "classification_accuracy",
        "score": score,
        "comment": f"Expected: {expected_type}, Got: {actual_type}"
    }

def check_behavior(run, example, client=None) -> dict:
    expected_behavior = example.outputs.get("expected_behavior")

    if run.outputs.get("final_response"):
        actual_behavior = "direct_answer"
    elif run.outputs.get("history") and len(run.outputs["history"]) > 0:
        actual_behavior = "full_debate"
    else:
        actual_behavior = "unknown"
    
    score = 1.0 if actual_behavior == expected_behavior else 0.0
    
    return {
        "key": "behavior_correctness",
        "score": score,
        "comment": f"Expected: {expected_behavior}, Got: {actual_behavior}"
    }

def check_debate_completeness(run, example, client=None) -> dict:
    if example.outputs.get("expected_behavior") != "full_debate":
        return {"key": "debate_completeness", "score": None, "comment": "N/A - not a debate"}
    
    history = run.outputs.get("history", [])
    num_rounds = run.outputs.get("num_rounds", 0)

    has_agent_a = sum(1 for h in history if "Agent A:" in h)
    has_agent_b = sum(1 for h in history if "Agent B:" in h)
    has_judge = any("JUDGE" in h for h in history)

    score = 0.0
    comments = []

    if has_agent_a >= 3:
        score += 0.3
        comments.append("✓ Agent A argued 3 rounds")
    else:
        comments.append(f"✗ Agent A only argued {has_agent_a} rounds")

    if has_agent_b >= 3:
        score += 0.3
        comments.append("✓ Agent B argued 3 rounds")
    else:
        comments.append(f"✗ Agent B only argued {has_agent_b} rounds")

    if has_judge:
        score += 0.4
        comments.append("✓ Judge provided verdict")
    else:
        comments.append("✗ No judge verdict found")

    return {
        "key": "debate_completeness",
        "score": score,
        "comment": " | ".join(comments)
    }

def create_debate_quality_evaluator():
    criteria = """
    Score the debate quality on a scale of 0-1 based on:
    - Coherence of arguments
    - Relevance
    - Judge fairness
    - Substantiveness
    """

    return LangChainStringEvaluator(
        "labeled_score_string",
        config={
            "criteria": {"debate_quality": criteria},
            "normalize_by": 10.0,
            "llm": get_llm()
        },
        prepare_data=lambda run, example: {
            "prediction": "\n".join(run.outputs.get("history", [])),
            "reference": f"Topic: {example.inputs['topic']}\nExpected: High-quality debate",
            "input": example.inputs["topic"]
        }
    )

def run_evaluation():
    print("🚀 Starting LangSmith Evaluation...\n")

    results = evaluate(
        run_debate_agent,
        data="debate-agent-evaluation",
        evaluators=[
            check_classification,
            check_behavior,
            check_debate_completeness,
            create_debate_quality_evaluator(),
        ],
        experiment_prefix="debate-agent",
        description="Evaluating debate agent",
        max_concurrency=2,
    )

    print("\n✅ Evaluation Complete!\n")
    print(f"- Experiment: {results.experiment_name}")

    return results

if __name__ == "__main__":
    run_evaluation()