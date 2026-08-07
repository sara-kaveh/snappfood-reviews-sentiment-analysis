from sklearn.model_selection import train_test_split
import pandas as pd
from data_preprocessing import DataPreprocessor
from models import TextClassifier
from train import Trainer
from config import Config


def main():

    preprocessor = DataPreprocessor()

    df = preprocessor.load_data(
        Config.DATASET_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=Config.RANDOM_SEED,
        stratify=df["label"])

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=Config.RANDOM_SEED,
        stratify=train_df["label"])

    x_train, y_train = preprocessor.fit_transform(
        train_df)

    x_val, y_val = preprocessor.transform(
        val_df)

    x_test, y_test = preprocessor.transform(
        test_df)

    model = TextClassifier.build(
        vocab_size=min(
            Config.MAX_FEATURES,
            len(preprocessor.tokenizer.word_index) + 1), num_classes=2)

    trainer = Trainer(model)

    trainer.train(
        x_train,
        y_train,
        x_val,
        y_val)

    trainer.evaluate(
        x_test,
        y_test,
        preprocessor.label_encoder)


if __name__ == "__main__":
    main()
