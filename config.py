class Config:

    DATASET_PATH = "data/cleaned_snappfood.csv"

    MAX_FEATURES = 15000

    MAX_LENGTH = 40

    EMBEDDING_DIM = 128

    LSTM_UNITS = 128

    BATCH_SIZE = 64

    EPOCHS = 10

    LEARNING_RATE = 0.001

    MODEL_PATH = "models/best_model.keras"

    TOKENIZER_PATH = "models/tokenizer.pkl"

    LABEL_ENCODER_PATH = "models/label_encoder.pkl"

    RANDOM_SEED = 42
