import logging
from pathlib import Path

logger = logging.getLogger("pipeline_tasks")


def run_vision_analysis(executor, output_dir: str | Path) -> bool:
    from vision_analyzer import analyze_panels

    provider = executor.get_provider("vision_analysis") if executor else None
    if not provider:
        raise RuntimeError("Executor required for vision_analysis task")

    result = analyze_panels(
        panels_dir=str(output_dir),
        provider=provider,
        output_dir=str(output_dir),
    )
    if not result:
        logger.error("Vision analysis returned empty result")
        return False
    n = len(result.get("panels", []))
    logger.info(f"[VISION] {n} paneles analizados con Gemini Flash Lite")
    return True


def run_scene_group(output_dir: str | Path) -> bool:
    from scene_grouper import group_scenes

    vision_path = Path(output_dir) / "vision_detail.json"
    if not vision_path.is_file():
        logger.error("vision_detail.json no encontrado para scene_group")
        return False

    result = group_scenes(
        vision_detail_path=str(vision_path),
        output_dir=str(output_dir),
    )
    n = result.get("total_scenes", 0)
    logger.info(f"[SCENE] {n} escenas agrupadas")
    return True


def run_interpretation(executor, output_dir: str | Path) -> bool:
    from interpretation_engine import interpret_panels

    provider = executor.get_provider("interpretation") if executor else None
    if not provider:
        raise RuntimeError("Executor required for interpretation task")

    vision_path = Path(output_dir) / "vision_detail.json"
    scenes_path = Path(output_dir) / "scenes.json"

    result = interpret_panels(
        vision_detail_path=str(vision_path),
        scenes_path=str(scenes_path),
        provider=provider,
        output_dir=str(output_dir),
    )
    n = len(result.get("panels", []))
    logger.info(f"[INTERPRET] {n} paneles interpretados con DeepSeek")
    return True


def run_curador(executor, output_dir: str | Path) -> bool:
    from curador_v5 import select_scenes

    provider = executor.get_provider("curador") if executor else None
    if not provider:
        raise RuntimeError("Executor required for curador task")

    scenes_path = Path(output_dir) / "scenes.json"
    interp_path = Path(output_dir) / "interpretation.json"

    result = select_scenes(
        scenes_path=str(scenes_path),
        interpretation_path=str(interp_path),
        provider=provider,
        output_dir=str(output_dir),
    )
    n = result.get("num_selected_scenes", 0)
    logger.info(f"[CURADOR] {n} escenas seleccionadas con DeepSeek")
    return True


def run_guion(executor, output_dir: str | Path) -> bool:
    from script_generator_v3 import generate_script

    provider = executor.get_provider("guion") if executor else None
    if not provider:
        raise RuntimeError("Executor required for guion task")

    selected_path = Path(output_dir) / "selected_panels.json"
    interp_path = Path(output_dir) / "interpretation.json"
    vision_path = Path(output_dir) / "vision_detail.json"

    result = generate_script(
        selected_panels_path=str(selected_path),
        interpretation_path=str(interp_path),
        vision_detail_path=str(vision_path),
        provider=provider,
        output_dir=str(output_dir),
    )
    if not result.get("script"):
        logger.error("Script generation returned empty result")
        return False
    n_segs = len(result.get("segments", {}).get("segments", []))
    logger.info(f"[GUION] Guion generado con DeepSeek ({n_segs} segmentos)")
    return True


from pipeline_steps_base import PipelineStep, PipelineContext


class VisionAnalysisStep(PipelineStep):
    def __init__(self, executor):
        super().__init__("vision_analysis")
        self._executor = executor

    def validate_contract(self, context: PipelineContext) -> bool:
        manifest = Path(context.output_dir) / "panels.json"
        if not manifest.is_file():
            self.logger.error("panels.json no encontrado para vision_analysis")
            return False
        return True

    def should_skip(self, context: PipelineContext) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        return (Path(context.output_dir) / "vision_detail.json").is_file()

    def execute(self, context: PipelineContext) -> bool:
        if self.should_skip(context):
            self.logger.info("Saltando vision_analysis")
            return True
        return run_vision_analysis(executor=self._executor, output_dir=context.output_dir)


class SceneGroupStep(PipelineStep):
    def __init__(self):
        super().__init__("scene_group")

    def validate_contract(self, context: PipelineContext) -> bool:
        vd = Path(context.output_dir) / "vision_detail.json"
        if not vd.is_file():
            self.logger.error("vision_detail.json no encontrado para scene_group")
            return False
        return True

    def should_skip(self, context: PipelineContext) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        return (Path(context.output_dir) / "scenes.json").is_file()

    def execute(self, context: PipelineContext) -> bool:
        if self.should_skip(context):
            self.logger.info("Saltando scene_group")
            return True
        return run_scene_group(output_dir=context.output_dir)


class InterpretationStep(PipelineStep):
    def __init__(self, executor):
        super().__init__("interpretation")
        self._executor = executor

    def validate_contract(self, context: PipelineContext) -> bool:
        vd = Path(context.output_dir) / "vision_detail.json"
        sc = Path(context.output_dir) / "scenes.json"
        if not vd.is_file():
            self.logger.error("vision_detail.json no encontrado para interpretation")
            return False
        if not sc.is_file():
            self.logger.error("scenes.json no encontrado para interpretation")
            return False
        return True

    def should_skip(self, context: PipelineContext) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        return (Path(context.output_dir) / "interpretation.json").is_file()

    def execute(self, context: PipelineContext) -> bool:
        if self.should_skip(context):
            self.logger.info("Saltando interpretation")
            return True
        return run_interpretation(executor=self._executor, output_dir=context.output_dir)


class CuradorStep(PipelineStep):
    def __init__(self, executor):
        super().__init__("curador")
        self._executor = executor

    def validate_contract(self, context: PipelineContext) -> bool:
        sc = Path(context.output_dir) / "scenes.json"
        ip = Path(context.output_dir) / "interpretation.json"
        if not sc.is_file():
            self.logger.error("scenes.json no encontrado para curador")
            return False
        if not ip.is_file():
            self.logger.error("interpretation.json no encontrado para curador")
            return False
        return True

    def should_skip(self, context: PipelineContext) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        return (Path(context.output_dir) / "selected_panels.json").is_file()

    def execute(self, context: PipelineContext) -> bool:
        if self.should_skip(context):
            self.logger.info("Saltando curador")
            return True
        return run_curador(executor=self._executor, output_dir=context.output_dir)


class ScriptGeneratorStep(PipelineStep):
    def __init__(self, executor):
        super().__init__("guion")
        self._executor = executor

    def validate_contract(self, context: PipelineContext) -> bool:
        sp = Path(context.output_dir) / "selected_panels.json"
        ip = Path(context.output_dir) / "interpretation.json"
        vd = Path(context.output_dir) / "vision_detail.json"
        for f, name in [(sp, "selected_panels.json"), (ip, "interpretation.json"), (vd, "vision_detail.json")]:
            if not f.is_file():
                self.logger.error(f"{name} no encontrado para guion")
                return False
        return True

    def should_skip(self, context: PipelineContext) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        guion = Path(context.output_dir) / "guion.txt"
        segments = Path(context.output_dir) / "segments.json"
        return guion.is_file() and segments.is_file()

    def execute(self, context: PipelineContext) -> bool:
        if self.should_skip(context):
            self.logger.info("Saltando guion")
            return True
        return run_guion(executor=self._executor, output_dir=context.output_dir)
