from llm.model import get_llm

llm = get_llm()

# -------------------------
# Topic Classifier Node
# -------------------------
def topic_classifier_node(state):
    topic = state["topic"]

    prompt = f"""
    Classify the user's request as one of the following:
    - factual: can be answered directly without debate
    - debatable: involves opinions, values, ethics, or subjective reasoning
    - other

    Return only one word: factual, debatable, or other.

    User question: "{topic}"
    """

    result = llm.invoke(prompt).content.strip().lower()

    return {"topic_type": result}

# -------------------------
# Direct Answer Node
# -------------------------
def direct_answer_node(state):
    topic = state["topic"]

    answer = llm.invoke(f"Answer the following question directly: {topic}").content

    return {"final_response": answer}

# -------------------------
# Agent A Node
# -------------------------
def agent_a_node(state):
    round_num = state["round_number"]
    
    prompt = f"""
You are Agent A in Round {round_num} of a debate. Argue FOR the topic:

"{state['topic']}"

Agent B previously said:
{state['agent_b_last'] if state['agent_b_last'] else "Nothing yet - this is the first round."}

Respond with your strongest argument within 5 sentences.
"""
    response = llm.invoke(prompt).content
    return {
        "agent_a_last": response,
        "history": state["history"] + [f"[Round {round_num}] Agent A: {response}"]
    }

# -------------------------
# Agent B Node
# -------------------------
def agent_b_node(state):
    round_num = state["round_number"]
    
    prompt = f"""
You are Agent B in Round {round_num} of a debate. Argue AGAINST the topic:

"{state['topic']}"

Agent A previously said:
{state['agent_a_last']}

Respond with your strongest counter-argument within 5 sentences.
"""
    response = llm.invoke(prompt).content
    return {
        "agent_b_last": response,
        "history": state["history"] + [f"[Round {round_num}] Agent B: {response}"]
    }

# -------------------------
# Round Counter Node
# -------------------------
def round_counter_node(state):
    """Increments the round counter after each debate round."""
    return {"round_number": state["round_number"] + 1}

# -------------------------
# Judge Node
# -------------------------
def judge_node(state):
    transcript = "\n\n".join(state["history"])

    prompt = f"""
You are the Judge. The debate topic is:

"{state['topic']}"

The debate lasted {state['round_number']} rounds.

Debate Transcript:
{transcript}

Provide:
1. A fair summary of both positions
2. Strengths & weaknesses of each side
3. Your final verdict on who presented the stronger argument and why

Be thorough and analytical in your judgment.
"""

    response = llm.invoke(prompt).content
    return {"history": state["history"] + [f"\n{'='*50}\nJUDGE'S FINAL VERDICT:\n{'='*50}\n{response}"]}