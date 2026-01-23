# Chicken Disease Classification - AI Agent Instructions

## Project Overview
CNN-based ML pipeline classifying chicken diseases (4 classes: Coccidiosis, Healthy, New Castle Disease, Salmonella) using VGG16 transfer learning. Config-driven modular architecture with strictly enforced workflow patterns.

## Architecture & Data Flow

### Core Module: `src/cnnClassifier/`
```
entity/config_entity.py      → Frozen dataclasses (typed configs)
config/configuration.py      → ConfigurationManager (YAML loader + config factory)
components/<stage>.py        → Business logic (data processing, model ops)
pipeline/stage_0X_<name>.py  → Stage orchestrators (wires config → component)
utils/common.py              → Shared utilities (read_yaml, save_json, create_directories)
constants/__init__.py        → Path constants (CONFIG_FILE_PATH, PARAMS_FILE_PATH)
__init__.py                  → Logger setup (file + stdout)
```

### Configuration System (3-Tier)
1. **config/config.yaml** - Stage paths & data sources (`artifacts_root`, download URLs)
2. **params.yaml** - Model hyperparameters (IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, etc.)
3. **Entity Classes** - Type-safe dataclasses mapping YAML → Python objects

**Data Flow**: `ConfigurationManager()` → `read_yaml()` → `create_directories()` → frozen dataclass → Component

## Development Workflow (9-Step Process)

**ALWAYS follow this sequence when adding new pipeline stages:**

1. **config/config.yaml** - Add stage section with `root_dir` and artifact paths
2. **secrets.yaml** [OPTIONAL] - Add sensitive keys (not tracked)
3. **params.yaml** - Add hyperparameters (prefix with `params_` in entity)
4. **entity/config_entity.py** - Create `@dataclass(frozen=True)` with `Path`/primitive types
5. **config/configuration.py** - Add `get_<stage>_config()` method returning entity
6. **components/<stage>.py** - Implement component with `__init__(config)` + methods
7. **pipeline/stage_0X_<name>.py** - Create pipeline class with `main()` method
8. **main.py** - Import pipeline, wrap with logging pattern, execute in try/except
9. **dvc.yaml** - [TODO] Define stage dependencies/outputs

## Critical Code Patterns

### Mandatory Logging Wrapper (main.py)
```python
from cnnClassifier import logger

STAGE_NAME = "Your Stage Name"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = YourPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
```
**Logs to**: `logs/running_logs.log` (format: `[timestamp: level: module: message]`)

### Component Pattern
```python
class YourComponent:
    def __init__(self, config: YourConfig):
        self.config = config  # Store typed config
    
    def main_method(self):
        # Access via self.config.root_dir, self.config.params_epochs
        pass
    
    @staticmethod
    def helper(arg):  # Use static for reusable logic without config
        pass
```

### Pipeline Pattern
```python
class YourPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()  # Auto-loads YAMLs
        stage_config = config.get_your_stage_config()
        component = YourComponent(config=stage_config)
        component.main_method()
```

### ConfigurationManager Method Pattern
```python
def get_your_stage_config(self) -> YourConfig:
    config = self.config.your_stage  # Dot notation via ConfigBox
    create_directories([config.root_dir])  # Always create dirs first
    
    return YourConfig(
        root_dir=Path(config.root_dir),
        some_path=Path(config.some_path),
        params_epochs=self.params.EPOCHS  # params_ prefix for hyperparameters
    )
```

## Path Handling Rules
- **ALWAYS** use `pathlib.Path` (cast in entity constructors)
- Artifacts structure: `artifacts/<stage_name>/<files>` (auto-created by ConfigManager)
- Import constants: `from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH`
- YAML paths are **relative to project root** (where main.py lives)
- **Run all scripts from project root** to avoid import/path issues

## Implemented Pipeline (Current State)

### Stage 01: Data Ingestion
- Downloads 70MB ZIP from GitHub (chicken fecal images dataset)
- Extracts to `artifacts/data_ingestion/Chicken-fecal-images/` (4 class folders)
- Skips download if `local_data_file` exists
- **Component**: `DataIngestion.download_file()` → `extract_zip_file()`

### Stage 02: Prepare Base Model
- Loads VGG16 (weights='imagenet', include_top=False)
- Saves base: `artifacts/prepare_base_model/base_model.h5`
- Adds custom head: Flatten → Dense(4, softmax)
- Freezes VGG16 layers, compiles with SGD + categorical_crossentropy
- Saves updated: `base_model_updated.h5`
- **Component**: `PrepareBaseModel.get_base_model()` → `update_base_model()`

### Stage 03: Training (NOT in main.py - only Stages 01-02 run currently)
- **PrepareCallback**: TensorBoard + ModelCheckpoint (save_best_only)
- **Training**: Loads updated model, creates train/valid generators (80/20 split)
- Optional augmentation (rotation, flip, shift, shear, zoom) via `params.AUGMENTATION`
- Saves final model: `artifacts/training/model.h5`
- **Pipeline**: `ModelTrainingPipeline` (imported but NOT executed in main.py)

### Stage 04: Evaluation (IN PROGRESS - component exists, not in main.py)
- Located: `components/evaluation.py`, `pipeline/stage_04_evaluation.py`
- Loads trained model, evaluates on validation set (30% split)
- **BUG**: `model.evalaute()` → should be `evaluate()` (typo in line 43)
- **BUG**: `target_size=self.config.params_image_size[:1]` → should be `[:2]` (width, height)
- **BUG**: `"lose": self.score[0]` → should be `"loss"` in scores.json
- Saves scores to `scores.json` (loss, accuracy)

## Environment & Execution

### Virtual Environment
```powershell
chi\Scripts\activate          # Activate venv (Windows)
python main.py                # Run pipeline (only Stages 01-02 currently)
```

### Package Setup
- Editable install: `pip install -e .` (reads setup.py)
- `sys.path.append('src')` in main.py enables `from cnnClassifier import ...`
- **SRC_REPO**: `cnnClassifier` (package name in setup.py)

### Key Files
- **main.py** - Entry point (adds `src/` to path, runs Stage 01-02 only)
- **setup.py** - Package metadata (`find_packages(where="src")`)
- **requirements.txt** - Pins TensorFlow, Flask, DVC, includes `-e .`
- **dvc.yaml** - Has 1 stage defined but **contains typos** (cmd path, deps path, outs path all wrong)

## Known Bugs & Issues

1. **evaluation.py line 43**: `model.evalaute()` → `model.evaluate()`
2. **evaluation.py line 20**: `target_size[:1]` → should be `[:2]` for (width, height)
3. **evaluation.py line 46**: `"lose"` → `"loss"` in JSON key
4. **configuration.py line 77**: `get_trainig_config()` → typo in method name (should be `training`)
5. **main.py**: Stage 03 (Training) pipeline imported but **never executed**
6. **dvc.yaml**: Contains typos in paths (`stages_01` → `stage_01`, `Checken` → `Chicken`)
7. **Model format**: Uses `.h5` (legacy) - consider SavedModel for TF 2.x compatibility

## Tech Stack
- **ML**: TensorFlow 2.x (Keras API, VGG16, ImageDataGenerator)
- **Config**: PyYAML, python-box (ConfigBox for dot notation)
- **Utils**: joblib, tqdm, pathlib
- **Web**: Flask + Flask-Cors (templates/index.html exists but not integrated)
- **MLOps**: DVC installed (dvc.yaml partially defined but broken)

## Common Pitfalls

1. **Import Errors**: Always run from project root (where config/ folder is)
2. **Frozen Dataclasses**: Entities are immutable - create new instance to "modify"
3. **ConfigBox Behavior**: Empty YAML raises `BoxValueError`
4. **Params Prefix**: Hyperparameters in entity MUST start with `params_` (e.g., `params_epochs`)
5. **Path Types**: YAML paths are strings - cast to `Path()` in entity constructors
6. **Method Naming**: ConfigManager methods: `get_<stage>_config()` (not `get_<stage>_configuration()`)

## Next Development Steps (Based on Current State)

1. **Fix evaluation.py bugs** (typos in lines 20, 43, 46)
2. **Add Stage 03 execution to main.py** (Training pipeline imported but not called)
3. **Add Stage 04 execution to main.py** (Evaluation pipeline exists)
4. **Fix dvc.yaml typos** and test DVC pipeline
5. **Integrate Flask web UI** (templates/index.html + prediction endpoint)

