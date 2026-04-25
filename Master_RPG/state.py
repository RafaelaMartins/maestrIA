from typing import Annotated, TypedDict, List, Dict, Any, Optional
import operator

# Reducer functions to accumulate lists in the state
def add_messages(existing: List[dict], new: List[dict]) -> List[dict]:
    if new is None:
        return existing
    return existing + new

class GameState(TypedDict):
    # Core Game Loop
    session_id: str
    messages: Annotated[List[dict], add_messages] # History of the game (role, content)
    current_input: str # User's latest action
    
    # Character Info
    char_name: str
    char_data: dict # Attributes, class, etc.
    
    # Game Context
    current_location: str
    active_npcs: dict # ID -> NPC details
    
    # Hidden Plots (Roteirista)
    world_lore: str
    main_goal: str
    current_plot_stage: int
    
    # Internal Routing State
    next_node: str # Tells the orchestrator where to go next
    requires_roll: bool # Flag if a test is needed
    roll_details: dict # Skill, DC, Attr
    roll_result: Optional[dict] # Outcome of the roll
