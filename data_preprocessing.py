import pandas as pd
import numpy as np
import joblib
import os
from hazm import Normalizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config


class DataPreprocessor:

    def __init__(self):
        self.tokenizer = None
        self.normalizer = Normalizer()

    def load_data(self, file_path):

        df = pd.read_csv(file_path)

        df = df[["comment_cleaned", "label"]]

        df = df.rename(columns={"comment_cleaned": "text"})

        df = df.dropna()

        return df

    def fit_transform(self, df):

        texts = [
            self.normalizer.normalize(str(text))
            for text in df["text"]
        ]

        y = df["label"].astype(int).to_numpy()

        self.tokenizer = Tokenizer(
            num_words=Config.MAX_FEATURES,
            oov_token="<OOV>"
        )

        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(
            texts
        )

        x = pad_sequences(
            sequences,
            maxlen=Config.MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        return np.array(x), np.array(y)

    def transform(self, df):

        texts = [
            self.normalizer.normalize(str(text))
            for text in df["text"]
        ]

        y = df["label"].astype(int).to_numpy()

        sequences = self.tokenizer.texts_to_sequences(
            texts
        )

        x = pad_sequences(
            sequences,
            maxlen=Config.MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        return np.array(x), np.array(y)

    def save_preprocessors(self):

        os.makedirs("models", exist_ok=True)

        joblib.dump(
            self.tokenizer,
            Config.TOKENIZER_PATH
        )

        print("Tokenizer saved successfully.")
