import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger)
from visualization import (save_training_history,
                           save_confusion_matrix, save_classification_report)
from config import Config


class Trainer:

    def __init__(self, model):
        self.model = model

    def train(self, x_train, y_train, x_val, y_val):

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
                restore_best_weights=True,
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=0.000001
            ),

            ModelCheckpoint(
                Config.MODEL_PATH,
                save_best_only=True,
                mode="min",
                monitor="val_loss"
            ),

            CSVLogger('results/training_log.csv', append=True)

        ]

        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks
        )

        save_training_history(history)

        return history

    def evaluate(self, x_test, y_test, label_encoder):

        predictions = self.model.predict(
            x_test,
            verbose=0
        )

        y_pred = np.argmax(
            predictions,
            axis=1
        )

        print("\nClassification Report")
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Negative", "Positive"],
            zero_division=0
        )
        print(report)

        save_classification_report(report)

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        save_confusion_matrix(y_test, y_pred, ["Negative", "Positive"])
