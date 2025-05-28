# milcontrol
Military access control demo using a Keras autoencoder.

## Setup

Run `install.bat` on Windows to install TensorFlow and other Python
dependencies. Then train the model and start the GUI:

```bat
python src\train_autoencoder.py
python src\guard_access_gui.py
```

Model files are stored in the `model` directory as base64 text
(`autoencoder.b64`, `encoder.b64`) along with `threshold.json`.
