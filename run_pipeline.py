import json
import logging
import os
import sys
from pathlib import Path

from cli_validator import parse_and_validate_args
from structured_logger import setup_structured_logging
from graceful_shutdown import shutdown_handler
from exceptions import ConfigurationError

from pipeline_steps_base import PipelineContext, PipelineStep
from panel_crop import CropPanelsStep
from generate_audio import GenerateAudioStep
from assemble_video import AssembleVideoStep

from pipeline_tasks import (
    VisionAnalysisStep, SceneGroupStep, InterpretationStep,
    CuradorStep, ScriptGeneratorStep,
)
from llm.task_scheduler import TaskScheduler
from llm.pipeline_state import StateManager

logger = logging.getLogger("pipeline")


def build_steps(args, executor) -> list[PipelineStep]:
    """Pipeline v5 (8 pasos): crop → vision → scene → interpret → select → script → audio → video.

    Orden completo:
      1. crop_panels: extrae viñetas del PDF
      2. vision_analysis: Gemini Flash Lite analiza cada panel (objetivo, sin interpretar)
      3. scene_group: agrupa paneles por cambio de escena (placeholder)
      4. interpretation: DeepSeek interpreta narrativamente los datos visuales
      5. curador: DeepSeek selecciona 15-25 escenas para narracion
      6. script_generator: DeepSeek genera guion + segmentacion
      7. generate_audio: TTS continuo desde el guion
      8. assemble_video: montaje por segmentos con crossfade
    """
    steps: list[PipelineStep] = [
        CropPanelsStep(),
        VisionAnalysisStep(executor),
        SceneGroupStep(),
        InterpretationStep(executor),
        CuradorStep(executor),
        ScriptGeneratorStep(executor),
        GenerateAudioStep(),
    ]
    music_base = getattr(args, 'music_base_dir', None)
    max_workers = getattr(args, 'max_workers', 4)
    target_duration = getattr(args, 'target_duration', 600.0)
    steps.append(AssembleVideoStep(
        resolution=args.resolution, fps=args.fps, crf=args.crf,
        preset=args.preset, encoder=args.encoder,
        music=str(args.music) if args.music else None,
        music_base_dir=str(music_base) if music_base else None,
        music_volume=args.music_volume,
        crossfade=args.crossfade,
        max_workers=max_workers,
        target_duration=target_duration,
    ))
    return steps


def _load_config(args):
    providers_config = getattr(args, 'providers_config', None) or str(Path(__file__).resolve().parent / 'config' / 'providers.cloud.yaml')
    config_path = Path(providers_config)
    if config_path.exists():
        import yaml
        return yaml.safe_load(config_path.read_text())
    return {}


def _create_executor(args):
    providers_config = getattr(args, 'providers_config', None) or str(Path(__file__).resolve().parent / 'config' / 'providers.cloud.yaml')
    config_path = Path(providers_config)
    if config_path.exists():
        from llm.task_executor import TaskExecutor
        executor = TaskExecutor(config_path=str(config_path))
        executor.warmup()
        return executor
    return None


def _create_scheduler(args):
    raw = _load_config(args)
    sched_cfg = raw.get("scheduler", {})
    return TaskScheduler(
        workers=sched_cfg.get("workers", 4),
        queue_size=sched_cfg.get("queue_size", 100),
    )


def _create_state_manager(args):
    raw = _load_config(args)
    state_path = args.output_dir / "pipeline_state.json"
    state_cfg = raw.get("state", {})
    return StateManager(
        state_path=state_path,
        autosave=state_cfg.get("autosave", True),
        interval_seconds=state_cfg.get("interval_seconds", 30.0),
    )





def run_pipeline(context: PipelineContext, steps: list[PipelineStep], executor, args) -> bool:
    """Ejecuta las 8 fases del pipeline en orden secuencial.

    Cada paso decide si debe ejecutarse o saltarse (output existente + no --force)
    a traves de should_skip().
    """
    scheduler = _create_scheduler(args)
    if executor:
        executor._scheduler = scheduler
    state_manager = _create_state_manager(args)
    state_manager.start(args.project)
    success = False

    try:
        for i, step in enumerate(steps, 1):
            if shutdown_handler.is_shutdown_requested():
                logger.warning("Shutdown solicitado. Deteniendo pipeline.")
                state_manager.complete(False)
                return False

            logger.info(f"[{i}/{len(steps)}] {step.name}")

            try:
                if not step.validate_contract(context):
                    logger.error(f"Contrato invalido para '{step.name}'. Abortando.")
                    state_manager.complete(False)
                    return False

                if getattr(args, 'force', False):
                    context.skip_phases = []
                    context.force = True

                if step.execute(context):
                    logger.info(f"Fase '{step.name}' completada.")
                    state_manager.checkpoint(node_id=step.name, task=step.name, status="done")
                else:
                    logger.error(f"Fase '{step.name}' fallo.")
                    state_manager.complete(False)
                    return False
            except Exception as e:
                logger.error(f"Error en fase '{step.name}': {e}", exc_info=True)
                state_manager.complete(False)
                return False

        logger.info("Pipeline completado.")
        success = True
    finally:
        if executor:
            executor.close()
            summary = executor.get_summary()
            if summary:
                print()
                print(summary)
        scheduler.shutdown()
        state_manager.complete(success)
        state_manager.shutdown()

    if not success:
        return False

    # 4. Post-procesamiento: Quality Auditor (opcional)
    audit_enabled = getattr(args, 'quality_audit', False)
    if audit_enabled:
        logger.info("--- Fase: quality_audit (validacion narrativa) ---")
        try:
            from quality_auditor import audit_segments

            if executor:
                audit_provider = executor.get_provider("auditor")
            else:
                from llm.factory import LLMProviderFactory
                audit_provider = LLMProviderFactory.create(
                    getattr(args, 'text_provider', 'gemini'),
                    model=getattr(args, 'text_model', 'gemini-flash-lite-latest'),
                    timeout=60,
                )
            findings = audit_segments(
                panels_dir=context.output_dir,
                provider=audit_provider,
                auto_fix=getattr(args, 'quality_audit_fix', False),
            )
            if findings:
                logger.warning(f"[AUDITOR] {len(findings)} incongruencias encontradas. Revisa quality_report_narrativo.json")
            else:
                logger.info("[AUDITOR] Validacion superada: 0 incongruencias")
        except Exception as e:
            logger.error(f"[AUDITOR] Error: {e}", exc_info=True)

    # 5. Thumbnail God (opcional)
    thumbnail_enabled = getattr(args, 'thumbnail', False)
    if thumbnail_enabled:
        logger.info("--- Fase: thumbnail_god (generacion de miniatura) ---")
        try:
            from thumbnail_god import generate_thumbnail

            if executor:
                thumb_provider = executor.get_provider("thumbnail")
            else:
                from llm.factory import LLMProviderFactory
                thumb_provider = LLMProviderFactory.create(
                    getattr(args, 'text_provider', 'gemini'),
                    model=getattr(args, 'text_model', 'gemini-flash-lite-latest'),
                    timeout=30,
                )
            thumb_path = generate_thumbnail(
                panels_dir=context.output_dir,
                provider=thumb_provider,
            )
            logger.info(f"[THUMBNAIL] Miniatura: {thumb_path}")
        except Exception as e:
            logger.error(f"[THUMBNAIL] Error: {e}", exc_info=True)

    return True


def main():
    try:
        args = parse_and_validate_args()
    except ConfigurationError as e:
        print(f"Error de configuracion: {e}", file=sys.stderr)
        sys.exit(2)

    setup_structured_logging(log_dir=str(args.output_dir), log_level=logging.INFO)
    logger.info(f"Iniciando pipeline: {args.project}")

    shutdown_handler.register_cleanup(lambda: logger.info("Shutdown handler activo"))

    context = PipelineContext()
    context.project_name = args.project
    context.input_dir = str(args.input_dir)
    context.output_dir = str(args.output_dir)
    context.script_path = str(args.script) if args.script else str(args.output_dir / "guion.txt")
    context.resolution = args.resolution
    context.fps = args.fps
    context.parallel = args.parallel
    context.max_workers = args.max_workers
    context.target_duration = getattr(args, 'target_duration', 600.0)
    context.skip_phases = args.skip
    context.skip_low_confidence = getattr(args, 'skip_low_confidence', True)

    executor = _create_executor(args)
    steps = build_steps(args, executor)
    success = run_pipeline(context, steps, executor, args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
