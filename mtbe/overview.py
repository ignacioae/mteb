from __future__ import annotations

import logging
from typing import Any

from mtbe.abstasks.AbsTask import AbsTask
from mtbe.abstasks.TaskMetadata import TASK_TYPE, TASK_DOMAIN, MODALITIES

logger = logging.getLogger(__name__)


def create_task_list() -> list[type[AbsTask]]:
    """Create a list of all available task classes."""
    tasks_categories_cls = list(AbsTask.__subclasses__())
    tasks = [
        cls
        for cat_cls in tasks_categories_cls
        for cls in cat_cls.__subclasses__()
        if cat_cls.__name__.startswith("AbsTask")
    ]
    return tasks


def create_name_to_task_mapping() -> dict[str, type[AbsTask]]:
    """Create a mapping from task names to task classes."""
    tasks = create_task_list()
    return {cls.metadata.name: cls for cls in tasks if hasattr(cls, 'metadata')}


# Global task registry
TASKS_REGISTRY = create_name_to_task_mapping()


class MTEBTasks(tuple):
    """Container for MTEB tasks with additional functionality."""
    
    def __repr__(self) -> str:
        return "MTEBTasks" + super().__repr__()
    
    @property
    def languages(self) -> set[str]:
        """Return all languages from tasks."""
        langs = set()
        for task in self:
            for lg in task.languages:
                langs.add(lg)
        return langs
    
    def to_dict(self) -> list[dict[str, Any]]:
        """Convert tasks to list of dictionaries."""
        return [task.metadata_dict for task in self]
    
    def print_summary(self):
        """Print a summary of the tasks."""
        print(f"\nSelected {len(self)} tasks:")
        print("-" * 50)
        
        for task in self:
            modalities = ", ".join(task.metadata.modalities)
            languages = ", ".join(task.metadata.eval_langs)
            print(f"• {task.metadata.name}")
            print(f"  Type: {task.metadata.type}")
            print(f"  Modalities: {modalities}")
            print(f"  Languages: {languages}")
            print()


def get_tasks(
    languages: list[str] | None = None,
    domains: list[TASK_DOMAIN] | None = None,
    task_types: list[TASK_TYPE] | None = None,
    tasks: list[str] | None = None,
    modalities: list[MODALITIES] | None = None,
    exclude_aggregate: bool = False,
) -> MTEBTasks:
    """Get a list of tasks based on the specified filters.

    Args:
        languages: A list of 3-letter language codes (ISO 639-3, e.g. "eng").
        domains: A list of task domains.
        task_types: A list of task types.
        tasks: A list of specific task names to include.
        modalities: A list of modalities to include.
        exclude_aggregate: If True, exclude aggregate tasks.

    Returns:
        A MTEBTasks object containing the filtered tasks.

    Examples:
        >>> get_tasks(tasks=["SampleImageTextRetrieval"])
        >>> get_tasks(task_types=["Retrieval"], modalities=["text", "image"])
        >>> get_tasks(languages=["eng"], domains=["Academic"])
    """
    if tasks:
        # Get specific tasks by name
        _tasks = []
        for task_name in tasks:
            if task_name in TASKS_REGISTRY:
                task_cls = TASKS_REGISTRY[task_name]
                _tasks.append(task_cls())
            else:
                logger.warning(f"Task '{task_name}' not found in registry")
        return MTEBTasks(_tasks)
    
    # Get all available tasks
    _tasks = [cls() for cls in create_task_list() if hasattr(cls, 'metadata')]
    
    # Apply filters
    if languages:
        _tasks = [t for t in _tasks if any(lang in t.metadata.eval_langs for lang in languages)]
    
    if domains:
        _tasks = [t for t in _tasks if t.metadata.domains and any(domain in t.metadata.domains for domain in domains)]
    
    if task_types:
        _tasks = [t for t in _tasks if t.metadata.type in task_types]
    
    if modalities:
        _tasks = [t for t in _tasks if any(mod in t.metadata.modalities for mod in modalities)]
    
    if exclude_aggregate:
        _tasks = [t for t in _tasks if not t.is_aggregate]
    
    return MTEBTasks(_tasks)


def get_task(task_name: str) -> AbsTask:
    """Get a specific task by name.

    Args:
        task_name: The name of the task to fetch.

    Returns:
        An initialized task object.

    Examples:
        >>> get_task("SampleImageTextRetrieval")
    """
    if task_name not in TASKS_REGISTRY:
        available_tasks = list(TASKS_REGISTRY.keys())
        raise KeyError(f"Task '{task_name}' not found. Available tasks: {available_tasks}")
    
    task_cls = TASKS_REGISTRY[task_name]
    return task_cls()


def list_available_tasks() -> list[str]:
    """List all available task names."""
    return list(TASKS_REGISTRY.keys())


def get_task_info(task_name: str) -> dict[str, Any]:
    """Get detailed information about a task."""
    task = get_task(task_name)
    return task.metadata_dict


def print_available_tasks():
    """Print information about all available tasks."""
    tasks = get_tasks()
    
    print(f"\nAvailable Tasks ({len(tasks)}):")
    print("=" * 50)
    
    # Group by type
    by_type = {}
    for task in tasks:
        task_type = task.metadata.type
        if task_type not in by_type:
            by_type[task_type] = []
        by_type[task_type].append(task)
    
    for task_type, type_tasks in by_type.items():
        print(f"\n{task_type} ({len(type_tasks)} tasks):")
        print("-" * 30)
        
        for task in type_tasks:
            modalities = ", ".join(task.metadata.modalities)
            languages = ", ".join(task.metadata.eval_langs)
            print(f"  • {task.metadata.name}")
            print(f"    Modalities: {modalities}")
            print(f"    Languages: {languages}")
            if task.metadata.description:
                print(f"    Description: {task.metadata.description}")
            print()


# Import tasks to populate the registry
def _import_tasks():
    """Import all task modules to populate the registry."""
    try:
        # Import task modules here
        import mtbe.tasks.sample_image_text_retrieval
    except ImportError:
        # Tasks not yet implemented
        pass


# Initialize the registry
_import_tasks()
