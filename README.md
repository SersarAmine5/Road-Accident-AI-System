# Accident Detection

To setup the python environment with the right libraries, execute in the project directory the following commands:

```bash
python -m venv .venv
```

```bash
# macos:
source .venv/bin/activate
# windows:
.\.venv\Scripts\activate
```

If the above command didn't work due to a **PowerShell execution policy error**, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in an administrator powershell and try again after restarting your terminal.

```bash
pip install -r requirements.txt
```

## Download the Data

```bash
# TODO: to be completed.
```

## Train a Model

```bash
python src/train.py
```

## Use the Pre-Trained Model

TODO: instructions on how to go to the YOLO directory, and get the best.pt model weights.

TODO: instructuons on how to move the weights to the models directory in order to be able to use them later in the streamlit thing.

TODO: instructions on how to run the streamlit project and use the pre-trained model there. Tell that we have example images and videos inside of the examples folder.

```bash
# TODO: to be completed.
```
