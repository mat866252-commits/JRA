from abc import ABC, abstractmethod
from typing import Optional, Any
import logging


class PipelineContext:
    def __init__(self, **kwargs):
        self.project_name: str = ""
        self.input_dir: str = ""
        self.output_dir: str = ""
        self.script_path: str = ""
        self.resolution: str = "1920x1080"
        self.fps: int = 30
        self.parallel: bool = False
        self.max_workers: int = 4
        self.target_duration: float = 600.0
        self.skip_phases: list[str] = []
        self.skip_low_confidence: bool = True
        self.batch_mode: bool = False
        self.state: dict = {}
        self.scene_groups: list = []
        self.script_lines: list = []
        self.matched_scenes: list = []
        self.final_video_path: Optional[str] = None
        self.error_message: Optional[str] = None
        self.validation_report: Optional[Any] = None
        self.critic_feedback: Optional[Any] = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_error(self, error: str):
        if self.error_message:
            self.error_message += f"\n{error}"
        else:
            self.error_message = error

    def get_scene_by_id(self, scene_id: str):
        return next((s for s in self.script_lines if isinstance(s, dict) and s.get("scene_id") == scene_id), None)


class PipelineStep(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{name}")

    @abstractmethod
    def execute(self, context: PipelineContext) -> bool:
        pass

    def validate_contract(self, context: PipelineContext) -> bool:
        return True

    def should_skip(self, context: PipelineContext) -> bool:
        return self.name in context.skip_phases

    def on_error(self, context: PipelineContext, error: Exception) -> bool:
        self.logger.error(f"Error en fase {self.name}: {error}", exc_info=True)
        return False
