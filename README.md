# Manhwa Summary Pipeline

Pipeline automatizado enterprise-grade para generar videos a partir de resumenes de manhwa/webtoons.

## Caracteristicas

- **Procesamiento automatico**: De paginas de manhwa a video final
- **Matching visual inteligente**: Usa modelos de vision (SigLIP, CLIP) para alinear escenas
- **Validacion narrativa**: Detecta problemas de continuidad, ritmo y coherencia
- **Critica con LLM**: Evaluacion automatica de calidad del guion (opcional)
- **Telemetria completa**: Metricas de rendimiento y dashboard interactivo
- **Escalable**: Soporta capitulos de 500+ escenas sin colapsar
- **Containerizado**: Docker con soporte CPU/GPU
- **CI/CD**: Pipeline automatizado con GitHub Actions
- **Seguro**: Gestion de secretos con .env, nunca hardcodeados
- **Robusto**: Escrituras atomicas, graceful shutdown, reintentos con backoff
- **Orquestacion en memoria**: Los modelos de IA se cargan UNA sola vez
- **Contratos estrictos**: Validacion de datos entre fases con Pydantic

## Requisitos

- Python 3.10+
- FFmpeg
- Docker (opcional, recomendado)
- GPU NVIDIA (opcional, para matching visual acelerado)

## Instalacion

### Opcion 1: Docker (Recomendado)

```bash
git clone https://github.com/mat866252-commits/JRA.git
cd JRA

cp .env.example .env

# docker compose up -d   # Docker (recomendado)
```

### Opcion 2: Instalacion local

```bash
git clone https://github.com/mat866252-commits/JRA.git
cd JRA

python -m venv venv
source venv/bin/activate

pip install -r requirements.colab.txt  # Colab
# pip install -r requirements.txt      # Local

python run_pipeline.py --project "Capitulo_001" \
    --input-dir ./input \
    --script guion.txt \
    --output-dir ./output
```

## Uso

### Comando basico

```bash
python run_pipeline.py \
    --project "Mi_Capitulo" \
    --input-dir ./input \
    --script guion.txt \
    --output-dir ./output \
    --resolution 1920x1080 \
    --fps 30
```

### Con matching visual (requiere GPU)

```bash
python run_pipeline.py \
    --project "Mi_Capitulo" \
    --input-dir ./input \
    --script guion.txt \
    --matcher-mode siglip \
    --parallel
```

### Con critica narrativa LLM

```bash
python run_pipeline.py \
    --project "Mi_Capitulo" \
    --input-dir ./input \
    --script guion.txt \
    --critic \
    --critic-model gpt-4o-mini
```

### Saltar fases

```bash
python run_pipeline.py \
    --project "Mi_Capitulo" \
    --input-dir ./input \
    --script guion.txt \
    --skip crop_panels verify_panels
```

### Argumentos disponibles

```
--project          Nombre del proyecto (requerido)
--input-dir        Directorio con paginas del manhwa (requerido)
--script           Ruta al archivo guion.txt (requerido)
--output-dir       Directorio de salida (default: output)
--resolution       Resolucion del video (default: 1920x1080)
--fps              FPS del video (default: 30)
--matcher-mode     Modo de matching: none|siglip|clip (default: none)
--parallel         Habilitar procesamiento paralelo
--max-workers      Numero de workers paralelos (default: 4)
--skip             Fases a saltar (ej: --skip crop_panels)
--validate         Ejecutar validacion narrativa
--critic           Ejecutar critica narrativa (LLM)
--critic-model     Modelo LLM para critica (default: gpt-4o-mini)
--dashboard        Generar dashboard HTML de metricas
--telemetry        Habilitar telemetria de rendimiento
```

## Output

El pipeline genera:

- `video_final.mp4` - Video final
- `telemetry.json` - Metricas completas del pipeline
- `dashboard.html` - Dashboard interactivo
- `pipeline.log` - Log estructurado JSON con rotacion
- `pipeline_state.json` - Estado para reanudacion

## Arquitectura

Ver [ARCHITECTURE.md](ARCHITECTURE.md) para detalles completos.

## Troubleshooting

Ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para problemas comunes.

## Testing

```bash
pytest tests/ -v
python test_scalability.py
```

## Formato del guion

El archivo `guion.txt` debe tener el formato:

```
ESCENA 0001: Narracion de la primera escena.
ESCENA 0002: Narracion de la segunda escena.
```

Cada escena debe estar numerada secuencialmente sin saltos.

## Google Colab

Ver [colab_setup.ipynb](colab_setup.ipynb) para ejecutar el pipeline en Colab.

```bash
# Comando basico en Colab con Gemini:
!python run_pipeline.py \
    --project "Capitulo_001" \
    --input-dir ./input \
    --output-dir ./output \
    --chapter 1 \
    --text-provider gemini \
    --text-model gemini-flash-lite-latest \
    --providers-config config/providers.cloud.yaml
```

## Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto esta bajo la Licencia MIT.