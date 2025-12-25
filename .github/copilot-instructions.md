# Chicken Disease Classification - AI Agent Instructions

## Project Overview
This is a modular ML pipeline for classifying chicken diseases (Coccidiosis vs Healthy) using CNN-based image classification with TensorFlow. The project follows a structured workflow with configuration-driven development.

## Architecture & Structure

### Core Module: `src/cnnClassifier/`
- **entity/**: Immutable dataclass configs (using `@dataclass(frozen=True)`)
- **config/**: ConfigurationManager orchestrates config loading from YAML files
- **components/**: Self-contained ML pipeline components (data ingestion, training, evaluation)
- **pipeline/**: Stage orchestrators that connect ConfigurationManager → Component
- **utils/common.py**: Shared utilities (YAML/JSON/binary I/O, path operations, base64 encoding)
- **constants/**: Path constants (`CONFIG_FILE_PATH`, `PARAMS_FILE_PATH`)

### Configuration System
Three-tier config approach:
- `config/config.yaml`: Paths and dataset URLs (artifacts, data sources)
- `params.yaml`: Model hyperparameters (IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE)
- Entity classes: Type-safe config objects created from YAML data

**Pattern**: ConfigurationManager reads YAMLs → creates directories → returns typed config entity

## Development Workflow (The Sacred 9 Steps)

Follow this exact sequence when adding new pipeline stages (e.g., model training, evaluation):

1. Update `config/config.yaml` with new stage paths/URLs
2. Update `secrets.yaml` [optional] for sensitive data
3. Update `params.yaml` with stage-specific hyperparameters
4. Create entity in `entity/config_entity.py` (frozen dataclass)
5. Add `get_<stage>_config()` method to ConfigurationManager
6. Create component class in `components/<stage>.py`
7. Create pipeline class in `pipeline/stage_<N>_<name>.py`
8. Add stage to `main.py` with try-except logger pattern
9. Update `dvc.yaml` for pipeline orchestration (currently empty)

## Key Conventions

### Logging Pattern
Always wrap pipeline stages with this exact structure:
```python
from cnnClassifier import logger

STAGE_NAME = "Your Stage Name"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    # Your code here
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
```

### Path Handling
- Use `pathlib.Path` for all file paths
- Artifact storage: `artifacts/<stage_name>/` (configured per stage)
- Constants defined in `src/cnnClassifier/constants/__init__.py`

### Component Pattern
Components are instantiated with config entities and expose public methods:
```python
class YourComponent:
    def __init__(self, config: YourConfig):
        self.config = config
    
    def main_method(self):
        # Implementation
```

### Pipeline Pattern
Pipelines orchestrate: ConfigurationManager → Component instantiation → Method calls
```python
class YourTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        stage_config = config.get_your_stage_config()
        component = YourComponent(config=stage_config)
        component.main_method()
```

## Essential Files

- **[template.py](template.py)**: Project scaffolding script (creates directories/empty files)
- **[main.py](main.py)**: Pipeline runner with `sys.path` modification for src imports
- **[setup.py](setup.py)**: Package metadata (SRC_REPO = "cnnClassifier")
- **[requirements.txt](requirements.txt)**: Note `-e.` for editable install

## Common Tasks

### Adding a New Pipeline Stage
1. Run through the 9-step workflow above
2. Test component independently before pipeline integration
3. Verify logging output follows the ">>>>>>...x==========x" pattern

### Data Ingestion (Current Implementation)
- Downloads ZIP from GitHub URL (configured in config.yaml)
- Extracts to `artifacts/data_ingestion/Chicken-fecal-images/`
- Checks file existence before re-downloading
- Uses `urllib.request` for downloads, `zipfile` for extraction

### Running the Project
```bash
python main.py  # Runs all registered pipeline stages
```

## Tech Stack
- **ML**: TensorFlow (CNN training), NumPy, Pandas, Scipy
- **Config**: PyYAML, python-box (ConfigBox for dot notation)
- **Utils**: joblib (binary serialization), tqdm (progress bars)
- **Serving**: Flask + Flask-Cors (for model API - templates exist)
- **MLOps**: DVC (data versioning - configured but not yet used)

## Important Notes
- Logs written to `logs/running_logs.log` + stdout
- Virtual environment in `chi/` directory (Windows: `chi\Scripts\activate`)
- Always use `Path` objects, not strings, for file operations
- ConfigBox enables dot notation access to YAML configs (`config.data_ingestion.root_dir`)
- `ensure_annotations` decorator in utils is a placeholder (returns func unchanged)

