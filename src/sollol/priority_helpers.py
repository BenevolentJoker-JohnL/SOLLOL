"""
Priority helpers for making SOLLOL's priority system more user-friendly.

SOLLOL uses a 10-level priority system (1=lowest, 10=highest) with age-based
fairness to prevent starvation. These helpers provide semantic names and
role-based priority mapping.

Example:
    from sollol.priority_helpers import Priority, get_priority_for_role

    # Use semantic constants
    pool.chat(model="llama3.2", messages=msgs, priority=Priority.HIGH)

    # Or map from agent roles
    priority = get_priority_for_role("researcher")  # Returns 8
    pool.chat(model="llama3.2", messages=msgs, priority=priority)
"""

from typing import Dict, Optional


# Priority Level Constants (1-10 scale, 10=highest)
class Priority:
    """Semantic priority levels for SOLLOL requests."""

    BATCH = 1  # Background batch processing, can wait indefinitely
    LOWEST = 2  # Very low priority, bulk operations
    LOW = 3  # Below normal, non-urgent tasks
    BELOW_NORMAL = 4  # Slightly below normal priority
    NORMAL = 5  # Default priority (baseline)
    ABOVE_NORMAL = 6  # Slightly above normal
    HIGH = 7  # Important, user-facing tasks
    HIGHER = 8  # Very important, interactive tasks
    URGENT = 9  # Critical tasks, fast response needed
    CRITICAL = 10  # Highest priority, mission-critical


# Agent Role → Priority Mapping
# Based on SynapticLlamas multi-agent orchestration patterns
AGENT_ROLE_PRIORITIES: Dict[str, int] = {
    # User-facing roles (high priority)
    "researcher": 8,  # Interactive research tasks
    "analyst": 8,  # User-requested analysis
    "assistant": 8,  # Direct user interaction
    "qa": 8,  # Quality assurance checks
    # Content processing roles (medium-high)
    "critic": 7,  # Critical analysis of outputs
    "reviewer": 7,  # Review and validation
    "editor": 6,  # Content editing and refinement
    # Background processing roles (medium)
    "summarizer": 5,  # Summarization tasks
    "classifier": 5,  # Classification tasks
    "extractor": 5,  # Information extraction
    # Batch processing roles (low)
    "indexer": 3,  # Batch indexing
    "crawler": 3,  # Web crawling
    "background": 2,  # General background tasks
    "batch": 1,  # Batch processing
}


# Task Type → Priority Mapping
TASK_TYPE_PRIORITIES: Dict[str, int] = {
    # Interactive tasks (high priority)
    "interactive": 9,  # Real-time user interaction
    "chat": 8,  # Chat/conversation
    "query": 8,  # User queries
    # Analysis tasks (medium-high)
    "analysis": 7,  # Analysis and reasoning
    "reasoning": 7,  # Complex reasoning tasks
    "generation": 6,  # Content generation
    # Processing tasks (medium)
    "summarization": 5,  # Summarization
    "classification": 5,  # Classification
    "extraction": 5,  # Information extraction
    "embedding": 4,  # Generate embeddings
    # Background tasks (low)
    "indexing": 3,  # Indexing operations
    "preprocessing": 3,  # Data preprocessing
    "batch": 1,  # Batch processing
}


def get_priority_for_role(role: str, default: int = Priority.NORMAL) -> int:
    """
    Get priority level for an agent role.

    Args:
        role: Agent role name (case-insensitive)
        default: Default priority if role not found

    Returns:
        Priority level (1-10)

    Example:
        priority = get_priority_for_role("researcher")  # Returns 8
        pool.chat(model="llama3.2", messages=msgs, priority=priority)
    """
    role_lower = role.lower().strip()
    return AGENT_ROLE_PRIORITIES.get(role_lower, default)


def get_priority_for_task(task_type: str, default: int = Priority.NORMAL) -> int:
    """
    Get priority level for a task type.

    Args:
        task_type: Task type name (case-insensitive)
        default: Default priority if task type not found

    Returns:
        Priority level (1-10)

    Example:
        priority = get_priority_for_task("interactive")  # Returns 9
        pool.chat(model="llama3.2", messages=msgs, priority=priority)
    """
    task_lower = task_type.lower().strip()
    return TASK_TYPE_PRIORITIES.get(task_lower, default)


def register_role_priority(role: str, priority: int) -> None:
    """
    Register a custom role → priority mapping.

    Args:
        role: Agent role name
        priority: Priority level (1-10)

    Raises:
        ValueError: If priority not in valid range

    Example:
        register_role_priority("custom_agent", 9)
    """
    if not 1 <= priority <= 10:
        raise ValueError(f"Priority must be 1-10, got {priority}")

    AGENT_ROLE_PRIORITIES[role.lower().strip()] = priority


def register_task_priority(task_type: str, priority: int) -> None:
    """
    Register a custom task type → priority mapping.

    Args:
        task_type: Task type name
        priority: Priority level (1-10)

    Raises:
        ValueError: If priority not in valid range

    Example:
        register_task_priority("custom_task", 7)
    """
    if not 1 <= priority <= 10:
        raise ValueError(f"Priority must be 1-10, got {priority}")

    TASK_TYPE_PRIORITIES[task_type.lower().strip()] = priority


class PriorityMapper:
    """
    Configurable priority mapper for custom priority schemes.

    Useful for complex multi-agent systems with custom priority logic.

    Example:
        mapper = PriorityMapper()
        mapper.add_role("custom_researcher", Priority.URGENT)
        mapper.add_task("realtime_chat", Priority.CRITICAL)

        priority = mapper.get_role_priority("custom_researcher")  # Returns 9
        pool.chat(model="llama3.2", messages=msgs, priority=priority)
    """

    def __init__(self):
        """Initialize with copies of default mappings."""
        self.role_map = AGENT_ROLE_PRIORITIES.copy()
        self.task_map = TASK_TYPE_PRIORITIES.copy()

    def add_role(self, role: str, priority: int) -> None:
        """Add or update a role priority mapping."""
        if not 1 <= priority <= 10:
            raise ValueError(f"Priority must be 1-10, got {priority}")
        self.role_map[role.lower().strip()] = priority

    def add_task(self, task_type: str, priority: int) -> None:
        """Add or update a task type priority mapping."""
        if not 1 <= priority <= 10:
            raise ValueError(f"Priority must be 1-10, got {priority}")
        self.task_map[task_type.lower().strip()] = priority

    def get_role_priority(self, role: str, default: int = Priority.NORMAL) -> int:
        """Get priority for a role."""
        return self.role_map.get(role.lower().strip(), default)

    def get_task_priority(self, task_type: str, default: int = Priority.NORMAL) -> int:
        """Get priority for a task type."""
        return self.task_map.get(task_type.lower().strip(), default)

    def remove_role(self, role: str) -> None:
        """Remove a role mapping."""
        self.role_map.pop(role.lower().strip(), None)

    def remove_task(self, task_type: str) -> None:
        """Remove a task type mapping."""
        self.task_map.pop(task_type.lower().strip(), None)

    def list_roles(self) -> Dict[str, int]:
        """Get all role mappings."""
        return self.role_map.copy()

    def list_tasks(self) -> Dict[str, int]:
        """Get all task type mappings."""
        return self.task_map.copy()


def explain_priority_system() -> str:
    """
    Get a human-readable explanation of SOLLOL's priority system.

    Returns:
        Formatted explanation string
    """
    return """
SOLLOL Priority System (1-10 scale)
====================================

Priority Levels:
  10 (CRITICAL)     - Mission-critical, highest priority
   9 (URGENT)       - Critical tasks, fast response needed
   8 (HIGHER)       - Very important, interactive tasks
   7 (HIGH)         - Important, user-facing tasks
   6 (ABOVE_NORMAL) - Slightly above normal
   5 (NORMAL)       - Default priority (baseline)
   4 (BELOW_NORMAL) - Slightly below normal
   3 (LOW)          - Below normal, non-urgent
   2 (LOWEST)       - Very low priority, bulk operations
   1 (BATCH)        - Background batch, can wait indefinitely

How It Works:
- Higher priority tasks execute first
- Age-based fairness prevents starvation
- Tasks gain priority boost as they wait longer
- System balances responsiveness vs fairness

Usage Examples:
  from sollol.priority_helpers import Priority

  # Use semantic constants
  pool.chat(..., priority=Priority.HIGH)

  # Or use role-based mapping
  priority = get_priority_for_role("researcher")
  pool.chat(..., priority=priority)

  # Or use task-based mapping
  priority = get_priority_for_task("interactive")
  pool.chat(..., priority=priority)
"""


# Convenience function to get all available role names
def list_available_roles() -> list:
    """Get list of all predefined agent roles."""
    return sorted(AGENT_ROLE_PRIORITIES.keys())


# Convenience function to get all available task types
def list_available_tasks() -> list:
    """Get list of all predefined task types."""
    return sorted(TASK_TYPE_PRIORITIES.keys())
