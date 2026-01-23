import tensorflow as tf
import numpy as np
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.20
        )

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:2],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self._valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset = "validation",
            shuffle = False,
            **dataflow_kwargs

        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self._valid_generator)
        self.calculate_metrics(model)

    def calculate_metrics(self, model):
        # Get predictions
        self._valid_generator.reset()
        predictions = model.predict(self._valid_generator, steps=len(self._valid_generator))
        y_pred = np.argmax(predictions, axis=1)
        y_true = self._valid_generator.classes
        
        # Class labels
        class_labels = list(self._valid_generator.class_indices.keys())
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrix = cm.tolist()
        
        # Classification Report
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(class_labels))
        )
        
        # Per-class metrics
        self.class_metrics = {}
        for idx, label in enumerate(class_labels):
            self.class_metrics[label] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx])
            }
        
        # Overall metrics
        self.overall_metrics = {
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1_score": float(np.mean(f1)),
            "weighted_precision": float(np.average(precision, weights=support)),
            "weighted_recall": float(np.average(recall, weights=support)),
            "weighted_f1_score": float(np.average(f1, weights=support))
        }
    
    def save_score(self):
        scores = {
            "loss": float(self.score[0]), 
            "accuracy": float(self.score[1]),
            "confusion_matrix": self.confusion_matrix,
            "class_labels": list(self._valid_generator.class_indices.keys()),
            "per_class_metrics": self.class_metrics,
            "overall_metrics": self.overall_metrics
        }
        save_json(path = Path("scores.json"), data = scores)


