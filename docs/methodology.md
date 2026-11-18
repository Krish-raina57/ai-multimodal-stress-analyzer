# Methodology reconstructed from the presentation

## 1. Text branch
- **Data described:** TwitterFull and Reddit samples.
- **Preprocessing described:** text cleaning, tokenization, and sequence padding.
- **Model described:** an embedding layer followed by an LSTM classifier.
- **Reported accuracy:** 88%.

The presentation does not preserve the exact vocabulary size, sequence length, class mapping, train/test split, or training configuration.

## 2. Audio branch
- **Data described:** RAVDESS speech dataset.
- **Feature extraction:** MFCCs, delta MFCCs, and double-delta MFCCs using Librosa.
- **Model described:** three convolutional layers, Global Average Pooling, and a dense classifier.
- **Reported accuracy:** 44%.

The presentation attributes the moderate result to overlap between general emotion labels and stress labels, as well as dataset imbalance.

## 3. Facial-expression branch
- **Data described:** FER2013.
- **Preprocessing:** grayscale conversion, resizing to 48×48 pixels, and normalization.
- **Model described:** CNN with Batch Normalization and Dropout.
- **Reported accuracy:** 52%.

The presentation identifies lighting, viewing angles, and occlusion as challenges.

## 4. Late fusion
The three unimodal models produce softmax probability outputs. The proposed fusion method averages those outputs to obtain a final prediction.

The presentation states that final evaluation was pending until all three datasets could be aligned. Simulated fusion tests were mentioned, but no final measured fusion accuracy was supplied.
