

## Installation

```bash
conda create -n code_classification python
conda activate code_classification
pip install -r requirements.txt
```
Training in a reasonable time requires a GPU
But testing can be done without.

### Running the model

```bash
python train_test.py
```

This implements a more sophisticated approach: fine-tuning BERT for text classification and tuning per-tag thresholds on validation to maximize F1-score.
It can be run with parameters (see description in the file) to either run training or testing and the fields used by the model.

Exemple:

```bash
python train_test.py --test_only --test_dataset ./path/to/your/test_data --checkpoint_path best_model.pth
```
This will run the train_test with the latest checkpoint on the test data


## Model Performance

The models are evaluated using the following metrics:
- Per-label precision, recall, and F1 score
- Overall weighted accuracy and F1 score
- Label-specific accuracy
