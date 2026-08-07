![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![BiLSTM](https://img.shields.io/badge/Model-BiLSTM-success)
[![Stars](https://img.shields.io/github/stars/sara-kaveh/snappfood-reviews-sentiment-analysis?style=social)](https://github.com/sara-kaveh/snappfood-reviews-sentiment-analysis)

# SnappFood Reviews Sentiment Analysis using Bidirectional LSTM

A deep learning project for **binary sentiment analysis of Persian user reviews** from the **SnappFood** dataset. It preprocesses Persian text, and trains a Bidirectional LSTM network. The final model achieves **86% test accuracy**.

---

## Features

* Persian text normalization using **Hazm**
* Tokenization with TensorFlow/Keras
* Bidirectional LSTM architecture
* Early Stopping
* ReduceLROnPlateau learning rate scheduling
* Best model checkpointing
* Saved tokenizer for inference
* Performance visualization

---

## Dataset

This project uses the **Cleaned SnappFood Persian Sentiment Analysis Dataset** available on Kaggle.

The dataset contains approximately **65,973** Persian user reviews collected from the SnappFood platform.

Each review is labeled as:

* Positive
* Negative

---

## Data Preprocessing

Before training, the dataset undergoes several preprocessing steps.

### Text Normalization

Persian text is normalized using **Hazm**, including:

* Character normalization
* Standardizing Persian/Arabic characters
* Removing unnecessary spaces
* Cleaning formatting inconsistencies

### Tokenization

The cleaned text is converted into integer sequences using the Keras Tokenizer.

Configuration:

* Vocabulary Size: **15,000**
* OOV token enabled

### Sequence Padding

Since reviews have different lengths, sequences are padded to a fixed size.

* Maximum sequence length: **40**
* Post-padding
* Post-truncation

---

## Model Architecture

<div align="center">

<pre>
Input Text
      │
Embedding (15000, 128)
      │
Bidirectional LSTM (128)
      │
Bidirectional LSTM (64)
      │
Dense (64, ReLU)
      │
Dropout (0.5)
      │
Dense (2, Softmax)
</pre>

</div>

---

## Hyperparameters

<div align="center">

| Parameter               |   Value |
| ----------------------- | ------: |
| Vocabulary Size         |  15,000 |
| Maximum Sequence Length |      40 |
| Embedding Dimension     |     128 |
| BiLSTM Units            | 128, 64 |
| Dense Units             |      64 |
| Dropout                 |     0.5 |
| Batch Size              |      64 |
| Learning Rate           |   0.001 |
| Epochs                  |      10 |
| Optimizer               |    Adam |

</div>

---

## Training Strategy

The model was trained using several techniques to improve convergence and reduce overfitting.

### Early Stopping

* Monitor: Validation Loss
* Patience: 5

### Learning Rate Scheduling

ReduceLROnPlateau automatically decreases the learning rate when validation performance stops improving.

### Model Checkpoint

The best-performing model is automatically saved according to validation loss.

---

## Results

### Test Performance

<div align="center">

| Metric          | Score |
| --------------- | :---: |
| Accuracy        | **86%** |
| Macro Precision | **86%** |
| Macro Recall    | **86%** |
| Macro F1-score  | **86%** |

</div>

---

### Classification Report

The trained model achieved balanced performance across both sentiment classes, demonstrating strong precision and recall for both positive and negative reviews.

<div align="center">

| Class | Precision | Recall | F1-score | Support |
|---------|---------:|------:|--------:|--------:|
| Negative | 0.84 | 0.88 | 0.86 | 4,845 |
| Positive | 0.88 | 0.84 | 0.86 | 5,051 |
| **Accuracy** | | | **0.86** | **9,896** |
| **Macro Avg** | **0.86** | **0.86** | **0.86** | **9,896** |
| **Weighted Avg** | **0.86** | **0.86** | **0.86** | **9,896** |

</div>

#### Observations

- The model achieved an overall **test accuracy of 86%**.
- Performance is well balanced across both sentiment classes, with identical **F1-scores (0.86)** for Positive and Negative reviews.
- The **Negative** class obtained a slightly higher **recall (0.88)**, indicating that the model successfully identifies most negative reviews.
- The **Positive** class achieved a higher **precision (0.88)**, meaning positive predictions are highly reliable.
- The similar macro and weighted averages suggest that the model generalizes consistently across the dataset without a strong bias toward either class.

---

### Training Curves

The repository includes:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

These plots illustrate the convergence behavior of the model throughout training.

<p align="center">
  <img src="results/training_curves_20260622_103135.png" width="1000">
</p>

---

### Learning Rate Schedule

The repository also includes the learning rate schedule generated during training.

<p align="center">
  <img src="results/lr_20260622_103135.png" width="600">
</p>

---

### Confusion Matrix

The confusion matrix provides a detailed view of prediction performance on the test set.

<p align="center">
  <img src="results/confusion_matrix_20260622_103148.png" width="600">
</p>

---

### Training Log

Training history is available, including:

<div align="center">

| Epoch | Train Accuracy | Train Loss | Learning Rate | Validation Accuracy | Validation Loss |
|------:|---------------:|-----------:|--------------:|--------------------:|----------------:|
| 0 | 83.21% | 0.3761 | 0.0010 | 85.83% | 0.3278 |
| 1 | 87.78% | 0.2946 | 0.0010 | 86.01% | 0.3387 |
| 2 | 89.58% | 0.2551 | 0.0010 | 85.44% | 0.3503 |
| 3 | 90.92% | 0.2233 | 0.0010 | 85.50% | 0.4016 |
| 4 | 93.01% | 0.1755 | 0.0005 | 85.16% | 0.4534 |
| 5 | 93.88% | 0.1551 | 0.0005 | 84.64% | 0.4979 |

</div>

---

## Example Predictions

<div align="center">

| Persian Review                     | Prediction |
| ---------------------------------- | ---------- |
| کیفیت غذا عالی بود                 | Positive   |
| ارسال خیلی دیر انجام شد            | Negative   |
| دوباره از این رستوران سفارش می‌دهم | Positive   |
| غذا سرد و بی‌کیفیت بود             | Negative   |

</div>

---

## Project Structure

```text
├── data/
│   └── cleaned_snappfood.csv
├── models/
│   └── best_model.keras
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── learning_rate_schedule.png
│   ├── training_curves.png
│   └── training_log.csv
├── config.py
├── data_preprocessing.py
├── main.py
├── models.py
├── train.py
├── predict.py
├── visualization.py
├── requirements.txt
└── README.md
```

---

## Author

**Sara Kaveh**

GitHub: https://github.com/sara-kaveh
