import os

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
# 1. Disable compression: Fixes the "403 Forbidden" error on some networks/proxies.
os.environ["LANGSMITH_DISABLE_RUN_COMPRESSION"] = "true"
# 2. Set Project Name: Ensures all traces go to this specific project in the UI.
os.environ["LANGCHAIN_PROJECT"] = "debate-agent-evaluation"

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from graph.build_graph import build_debate_graph
from llm.model import get_llm

client = Client()

# ------------------------------------------------------------------------------
# TARGET FUNCTION (The "Subject Under Test")
# ------------------------------------------------------------------------------
def run_debate_agent(inputs: dict) -> dict:
    """
    This is the wrapper function that LangSmith calls for every example in the dataset.
    It acts as a bridge between the Dataset (inputs) and your Graph (logic).
    """
    topic = inputs["topic"]
    
    # Initialize the default state for the graph
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
    
    # Build and run the graph
    graph = build_debate_graph()
    result = graph.invoke(initial_state)
    
    # Return a simplified dictionary that the Evaluators can easily read.
    # We strip out internal state (like 'agent_a_last') and keep only what we need to grade.
    return {
        "topic_type": result.get("topic_type"),     # Needed for check_classification
        "final_response": result.get("final_response"), # Needed for check_behavior
        "history": result.get("history", []),       # Needed for debate quality
        "num_rounds": result.get("round_number", 0) # Needed for completeness check
    }

# ------------------------------------------------------------------------------
# EVALUATOR 1: CLASSIFICATION ACCURACY
# ------------------------------------------------------------------------------
def check_classification(run, example, client=None) -> dict:
    """
    DETERMINISTIC EVALUATOR
    Checks if the Topic Classifier Node correctly identified the input type.
    """
    # 1. Get the "Ground Truth" from the dataset
    expected_type = example.outputs.get("expected_type")
    
    # 2. Get the "Actual Prediction" from the agent's run
    actual_type = run.outputs.get("topic_type")

    # 3. Binary Scoring: 1.0 (Correct) or 0.0 (Incorrect)
    score = 1.0 if actual_type == expected_type else 0.0
    
    return {
        "key": "classification_accuracy",
        "score": score,
        "comment": f"Expected: {expected_type}, Got: {actual_type}"
    }

# ------------------------------------------------------------------------------
# EVALUATOR 2: BEHAVIOR CORRECTNESS
# ------------------------------------------------------------------------------
def check_behavior(run, example, client=None) -> dict:
    """
    LOGIC/ROUTING EVALUATOR
    Checks if the Graph routed the user to the correct path (Debate vs Direct Answer).
    It does not read the text, it just looks at the structure of the output.
    """
    expected_behavior = example.outputs.get("expected_behavior")

    # Logic: Detect behavior based on which output keys are present
    if run.outputs.get("final_response"):
        # If 'final_response' exists, the DirectAnswerNode ran
        actual_behavior = "direct_answer"
    elif run.outputs.get("history") and len(run.outputs["history"]) > 0:
        # If 'history' has content, the Debate Nodes ran
        actual_behavior = "full_debate"
    else:
        actual_behavior = "unknown"
    
    # Compare calculated behavior vs expected behavior
    score = 1.0 if actual_behavior == expected_behavior else 0.0
    
    return {
        "key": "behavior_correctness",
        "score": score,
        "comment": f"Expected: {expected_behavior}, Got: {actual_behavior}"
    }

# ------------------------------------------------------------------------------
# EVALUATOR 3: DEBATE COMPLETENESS (Heuristic)
# ------------------------------------------------------------------------------
def check_debate_completeness(run, example, client=None) -> dict:
    """
    HEURISTIC EVALUATOR
    Checks if the debate followed the correct rules:
    - Did Agent A speak 3 times?
    - Did Agent B speak 3 times?
    - Did the Judge speak?
    """
    # Skip this check if it wasn't supposed to be a debate (e.g., factual questions)
    if example.outputs.get("expected_behavior") != "full_debate":
        return {"key": "debate_completeness", "score": None, "comment": "N/A - not a debate"}
    
    history = run.outputs.get("history", [])

    # Count how many times each actor appears in the history list
    has_agent_a = sum(1 for h in history if "Agent A:" in h)
    has_agent_b = sum(1 for h in history if "Agent B:" in h)
    has_judge = any("JUDGE" in h for h in history)

    score = 0.0
    comments = []

    # Weighted Scoring Logic:
    # 0.3 points for Agent A
    if has_agent_a >= 3:
        score += 0.3
        comments.append("✓ Agent A argued 3 rounds")
    else:
        comments.append(f"✗ Agent A only argued {has_agent_a} rounds")

    # 0.3 points for Agent B
    if has_agent_b >= 3:
        score += 0.3
        comments.append("✓ Agent B argued 3 rounds")
    else:
        comments.append(f"✗ Agent B only argued {has_agent_b} rounds")

    # 0.4 points for the Judge
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

# ------------------------------------------------------------------------------
# EVALUATOR 4: DEBATE QUALITY (LLM-as-a-Judge)
# ------------------------------------------------------------------------------
def create_debate_quality_evaluator():
    """
    LLM EVALUATOR
    Constructs an LLM Chain that reads the entire debate transcript and
    grades it based on subjective criteria (coherence, relevance).
    """
    criteria = """
    Score the debate quality on a scale of 0-1 based on:
    - Coherence of arguments
    - Relevance
    - Judge fairness
    - Substantiveness
    """

    return LangChainStringEvaluator(
        "labeled_score_string", # The internal type of evaluator to use
        config={
            # 'debate_quality_score' is the name of the column in the UI
            "criteria": {"debate_quality_score": criteria},
            # Normalize the LLM's 1-10 output to a 0-1 scale
            "normalize_by": 10.0,
            "llm": get_llm()
        },
        # Logic to prepare the inputs for the Judge LLM
        prepare_data=lambda run, example: {
            "prediction": "\n".join(run.outputs.get("history", [])),
            "reference": f"Topic: {example.inputs['topic']}\nExpected: High-quality debate",
            "input": example.inputs["topic"]
        }
    )

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def run_evaluation():
    print("🚀 Starting LangSmith Evaluation...\n")

    results = evaluate(
        run_debate_agent,              # The function to test
        data="debate-agent-evaluation",# The dataset to use
        evaluators=[                   # The list of grading functions
            check_classification,
            check_behavior,
            check_debate_completeness,
            create_debate_quality_evaluator(),
        ],
        experiment_prefix="debate-agent", # Prefix for the experiment name in UI
        description="Evaluating debate agent",
        max_concurrency=2,             # Run 2 tests at a time
    )

    print("\n✅ Evaluation Complete!\n")
    print(f"- Experiment: {results.experiment_name}")

    return results

if __name__ == "__main__":
    run_evaluation()