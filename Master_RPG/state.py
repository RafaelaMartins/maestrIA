from typing import TypedDict, List, Dict, Any, Optional

class GameState(TypedDict):
    # Core Game Loop
    session_id: str
    messages: List[dict] # History of the game (role, content)
    current_input: str # User's latest action
    turn_count: int # Tracks how many actions the player has taken
    
    # Character Info
    char_name: str
    char_data: dict # Attributes, class, etc.
    
    # Game Context
    current_location: str
    active_npcs: dict # ID -> NPC details
    interacting_npc: Optional[str] # Name or ID of the NPC currently being talked to
    
    # Hidden Plots (Roteirista)
    world_lore: str
    main_goal: str
    current_plot_stage: int
    
    # Internal Routing State
    next_node: str # Tells the orchestrator where to go next
    requires_roll: bool # Flag if a test is needed
    roll_details: dict # Skill, DC, Attr
    roll_result: Optional[dict] # Outcome of the roll
