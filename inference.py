import joblib
from predict import Predictor
from config import Config


def main():

    # Load tokenizer
    tokenizer = joblib.load(
        Config.TOKENIZER_PATH
    )

    # Initialize predictor
    predictor = Predictor(
        model_path=Config.MODEL_PATH,
        tokenizer=tokenizer
    )

    # Example reviews
    test_reviews = [
        "کیفیت غذا عالی بود و خیلی خوشمزه بود",
        "غذا سرد رسید و اصلا کیفیت خوبی نداشت",
        "ارسال خیلی سریع بود و از سفارش راضی هستم",
        "بدترین تجربه‌ای بود که داشتم، دوباره سفارش نمی‌دهم",
        "غذا معمولی بود، نه خوب نه بد"
    ]

    # Predict
    results = predictor.predict_batch(
        test_reviews
    )

    # Display results
    print("\nPrediction Results\n")
    print("-" * 50)

    for result in results:

        print(f"Text: {result['text']}")
        print(f"Prediction: {result['label']}")
        print(
            f"Confidence: {result['confidence']:.2%}"
        )
        print("-" * 50)


if __name__ == "__main__":
    main()
