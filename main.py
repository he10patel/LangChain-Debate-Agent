from graph.build_graph import build_debate_graph

def main():
    topic = input("Enter a debate topic: ").strip()

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

    print("\n" + "="*60)
    print("FINAL DEBATE OUTPUT")
    print("="*60 + "\n")
    
    # Check if it was a direct answer
    if result.get("final_response"):
        print(result["final_response"])
    else:
        # It was a debate - print the full transcript
        for line in result["history"]:
            print(line)
            print()  # Add spacing between entries

if __name__ == "__main__":
    main()