from typing import TypedDict, List, Optional

class DebateState(TypedDict):
    topic: str
    topic_type: Optional[str]
    history: List[str]
    agent_a_last: Optional[str]
    agent_b_last: Optional[str]
    continue_debate: bool
    final_response: Optional[str]
    round_number: int
    max_rounds: int