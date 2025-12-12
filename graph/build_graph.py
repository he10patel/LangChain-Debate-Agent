from langgraph.graph import StateGraph, START, END
from graph.state import DebateState
from graph.nodes import (
    topic_classifier_node,
    direct_answer_node,
    agent_a_node,
    agent_b_node,
    judge_node,
    round_counter_node
)

def build_debate_graph():
    graph = StateGraph(DebateState)

    # Register nodes
    graph.add_node("topic_classifier", topic_classifier_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("agent_a", agent_a_node)
    graph.add_node("agent_b", agent_b_node)
    graph.add_node("round_counter", round_counter_node)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "topic_classifier")

    # Conditional routing from topic classifier
    graph.add_conditional_edges(
        "topic_classifier",
        lambda state: state["topic_type"],
        {
            "factual": "direct_answer",
            "other": "direct_answer",
            "debatable": "agent_a"
        }
    )

    # Debate flow
    graph.add_edge("agent_a", "agent_b")
    graph.add_edge("agent_b", "round_counter")
    
    # Conditional routing: continue debate or go to judge
    graph.add_conditional_edges(
        "round_counter",
        lambda state: "continue" if state["round_number"] <= state["max_rounds"] else "judge",
        {
            "continue": "agent_a",
            "judge": "judge"
        }
    )
    
    graph.add_edge("judge", END)

    # Factual flow
    graph.add_edge("direct_answer", END)

    return graph.compile()