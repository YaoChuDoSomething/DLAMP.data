from abc import ABC, abstractmethod
from typing import List
from .context import Context
import logging

logging.basicConfig(level=logging.INFO, format="[PIPELINE] %(message)s")
logger = logging.getLogger(__name__)


class Task(ABC):
    """
    Abstract Base Class for all pipeline tasks.
    """

    @abstractmethod
    def execute(self, context: Context):
        pass


class Pipeline:
    """
    Orchestrates the execution of a strictly ordered list of tasks.
    """

    def __init__(self, context: Context):
        self.context = context
        self.tasks: List[Task] = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def run(self):
        logger.info("Starting Pipeline...")
        self.context.load_config()

        # We need to iterate over time steps typically
        # But for now, let's assume the timeline loop happens EITHER inside the pipeline
        # OR the pipeline is run per timestep.
        # Based on dlamp_prep.py, it loops over time.
        # Let's make the Pipeline run for the *entire* job,
        # but tasks might internally loop OR the pipeline exposes a method to run for a specific time.

        # Actually, to keep it clean:
        # The Orchestrator (dlamp_prep.py) builds the timeline and calls the pipeline or tasks.
        # BUT, to be truly OOP, the Pipeline could manage the timeline if configured.

        # Let's stick to the simplest flow: Pipeline executes tasks sequentially.
        # If a task needs to loop, it loops.
        # However, for Download -> Convert -> Regrid -> Diagnose, this is usually per-file/per-time.

        # Let's assume the Tasks handle the timeline IF they are batch tasks,
        # OR we create a "TimeLoopPipeline" if we want to granularly control it.
        # For this refactor, let's let the Pipeline simply run the tasks provided.

        for task in self.tasks:
            logger.info(f"Executing {task.__class__.__name__}...")
            task.execute(self.context)

        logger.info("Pipeline Completed.")
