"""
Create a test dataset for evaluating the debate agent.
This dataset contains various topics that should be classified and debated appropriately.
"""

from langsmith import Client

def create_debate_dataset():
    client = Client()
    
    # Define test examples with expected behaviors
    examples = [
        # Debatable topics - should trigger full debate
        {
            "topic": "Should artificial intelligence be regulated by governments?",
            "expected_type": "debatable",
            "expected_behavior": "full_debate"
        },
        {
            "topic": "Is remote work better than office work?",
            "expected_type": "debatable",
            "expected_behavior": "full_debate"
        },
        {
            "topic": "Should college education be free for all students?",
            "expected_type": "debatable",
            "expected_behavior": "full_debate"
        },
        {
            "topic": "Is social media doing more harm than good?",
            "expected_type": "debatable",
            "expected_behavior": "full_debate"
        },
        {
            "topic": "Should the voting age be lowered to 16?",
            "expected_type": "debatable",
            "expected_behavior": "full_debate"
        },
        
        # Factual questions - should give direct answers
        {
            "topic": "What is the capital of France?",
            "expected_type": "factual",
            "expected_behavior": "direct_answer"
        },
        {
            "topic": "How many planets are in our solar system?",
            "expected_type": "factual",
            "expected_behavior": "direct_answer"
        },
        {
            "topic": "What is the boiling point of water?",
            "expected_type": "factual",
            "expected_behavior": "direct_answer"
        },
        
        # Edge cases
        {
            "topic": "Why is the sky blue?",
            "expected_type": "factual",
            "expected_behavior": "direct_answer"
        },
        {
            "topic": "Is pineapple an acceptable pizza topping?",
            "expected_type": "debatable",
            "expected_behavior": "full_debate"
        }
    ]
    
    # Create dataset in LangSmith
    dataset_name = "debate-agent-evaluation"
    
    # Check if dataset exists, if so delete it
    try:
        client.delete_dataset(dataset_name=dataset_name)
        print(f"Deleted existing dataset: {dataset_name}")
    except:
        pass
    
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Test dataset for evaluating debate agent classification and debate quality"
    )
    
    # Add examples to dataset
    for example in examples:
        client.create_example(
            inputs={"topic": example["topic"]},
            outputs={
                "expected_type": example["expected_type"],
                "expected_behavior": example["expected_behavior"]
            },
            dataset_id=dataset.id
        )
    
    print(f"✅ Created dataset '{dataset_name}' with {len(examples)} examples")
    print(f"Dataset ID: {dataset.id}")
    return dataset

if __name__ == "__main__":
    create_debate_dataset()