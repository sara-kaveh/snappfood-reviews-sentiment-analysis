from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from config import Config


class TextClassifier:

    @staticmethod
    def build(vocab_size, num_classes):

        model = Sequential([

            Embedding(
                input_dim=vocab_size,
                output_dim=Config.EMBEDDING_DIM,
                input_length=Config.MAX_LENGTH),

            Bidirectional(
                LSTM(
                    Config.LSTM_UNITS,
                    return_sequences=True,
                    dropout=0.2)),

            Bidirectional(
                LSTM(
                    Config.LSTM_UNITS // 2,
                    dropout=0.2)),

            Dropout(0.5),

            Dense(
                64,
                activation="relu"),

            Dense(
                2,
                activation="softmax")
        ])

        model.compile(
            optimizer=Adam(
                learning_rate=Config.LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

        return model
