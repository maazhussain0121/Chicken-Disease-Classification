from cnnClassifier import logger
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline


STAGE_NAME = "Training"
try:
    logger.info(f"**************")
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e
