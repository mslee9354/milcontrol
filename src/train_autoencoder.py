import sys
import os

sys.path.append(os.path.dirname(__file__))
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import encode_info, INPUT_SIZE, LATENT_SIZE, PURPOSES, DESTINATIONS, TIMES


def build_model():
    inp = layers.Input(shape=(INPUT_SIZE,))
    x = layers.Dense(12, activation='relu')(inp)
    x = layers.Dense(10, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dense(6, activation='relu')(x)
    encoded = layers.Dense(LATENT_SIZE, activation='relu')(x)

    x = layers.Dense(6, activation='relu')(encoded)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dense(10, activation='relu')(x)
    x = layers.Dense(12, activation='relu')(x)
    out = layers.Dense(INPUT_SIZE, activation='sigmoid')(x)

    autoencoder = models.Model(inp, out)
    encoder = models.Model(inp, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def create_dataset():
    """Create simple synthetic normal access records."""
    sequences = {
        "24-760001": [
            ("근무", "본부", "주간"),
            ("근무", "본부", "주간"),
            ("근무", "본부", "주간"),
            ("근무", "본부", "야간"),
            ("근무", "본부", "주간"),
        ],
        "24-760002": [
            ("정비", "창고", "주간"),
            ("정비", "창고", "주간"),
            ("정비", "창고", "주간"),
            ("정비", "창고", "주간"),
            ("정비", "창고", "야간"),
        ],
        "23-760003": [
            ("복지", "병영", "주간"),
            ("복지", "병영", "주간"),
            ("복지", "병영", "주간"),
            ("복지", "병영", "주간"),
            ("복지", "병영", "주간"),
        ],
    }

    X = []
    for soldier, seq in sequences.items():
        prev = np.zeros(LATENT_SIZE)
        for purpose, dest, time in seq:
            info_vec = encode_info(purpose, dest, time)
            inp = np.concatenate([prev, info_vec])
            X.append(inp)
            prev = np.zeros(LATENT_SIZE)  # initial training uses zeros
    X = np.array(X)
    return X


def main():
    X = create_dataset()
    autoencoder, encoder = build_model()
    autoencoder.fit(X, X, epochs=50, batch_size=4, verbose=0)

    # calculate reconstruction error threshold
    preds = autoencoder.predict(X)
    mse = np.mean(np.square(preds - X), axis=1)
    threshold = float(np.mean(mse) + np.std(mse))

    import base64, tempfile

    def save_b64(model, path):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.h5')
        os.close(tmp_fd)
        model.save(tmp_path)
        with open(tmp_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        with open(path + '.b64', 'w', encoding='ascii') as f:
            f.write(b64)
        os.remove(tmp_path)

    os.makedirs('model', exist_ok=True)
    save_b64(autoencoder, 'model/autoencoder.h5')
    save_b64(encoder, 'model/encoder.h5')
    with open('model/threshold.json', 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold}, f, ensure_ascii=False, indent=2)

    print('Model and threshold saved.')


if __name__ == '__main__':
    main()


