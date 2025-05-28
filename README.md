# milcontrol

Simple guard post access control demo with anomaly detection.

## Components

- `src/train_autoencoder.py` – train the autoencoder using synthetic normal
  access data and save the model in base64 format under `model/`.
- `src/guard_access_gui.py` – Tkinter GUI to check access records using the
  pretrained model.
- `install.bat` – basic Windows script to install TensorFlow.

The model files `autoencoder.h5.b64` and `encoder.h5.b64` are base64-encoded
Keras models. They are decoded automatically at runtime.
