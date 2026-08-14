# CSCI544 Homework 3 – Named Entity Recognition (NER)

## 📌 Overview

This project implements Named Entity Recognition (NER) using Bidirectional LSTM-based models in PyTorch on the CoNLL-2003 dataset.

The following models are implemented:

* **Task 1:** BiLSTM with randomly initialized embeddings
* **Task 2:** BiLSTM with pretrained GloVe embeddings and capitalization handling
* **Bonus:** BiLSTM with character-level CNN

All experiments use **random seed = 42** for reproducibility, as required.

---

## 📁 Submitted Files

### Required Files

```id="reqfiles"
blstm1.pt        # Task 1 trained model
blstm2.pt        # Task 2 trained model

dev1.out         # Task 1 predictions (dev)
test1.out        # Task 1 predictions (test)

dev2.out         # Task 2 predictions (dev)
test2.out        # Task 2 predictions (test)

hw3_combined.py  # Source code
README.md        # This file
report.pdf       # Write-up
```

### Bonus Files

```id="bonusfiles"
bonus.pt         # Bonus model
pred             # Bonus test predictions
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* NumPy

Install dependencies:

```bash id="install"
pip install torch numpy
```

---

## 📂 Data Setup

Place the dataset in the following structure:

```id="datastructure"
data/
├── train
├── dev
├── test
```

Download and extract GloVe embeddings:

```bash id="glove"
gunzip glove.6B.100d.gz
```

---

## 🚀 How to Run

The script supports different modes: `task1`, `task2`, `bonus`, and `all`.

Ensure the paths in the commands are correct.

---

## 🔹 Task 1: Training + Prediction

### Train and generate predictions:

```bash id="task1cmd"
python hw3_combined.py \
  --mode task1 \
  --data-dir ./data \
  --output-dir .
```

Outputs:

* `blstm1.pt`
* `dev1.out`
* `test1.out`

---

## 🔹 Task 2: Training + Prediction (GloVe)

```bash id="task2cmd"
python hw3_combined.py \
  --mode task2 \
  --data-dir ./data \
  --glove-path ./glove.6B.100d.txt \
  --output-dir .
```

Outputs:

* `blstm2.pt`
* `dev2.out`
* `test2.out`

---

## 🔹 Bonus: Training + Prediction (Optional)

```bash id="bonuscmd"
python hw3_combined.py \
  --mode bonus \
  --data-dir ./data \
  --glove-path ./glove.6B.100d.txt \
  --output-dir .
```

Outputs:

* `bonus.pt`
* `pred`

---

## 🔹 Run All Models

```bash id="allcmd"
python hw3_combined.py \
  --mode all \
  --data-dir ./data \
  --glove-path ./glove.6B.100d.txt \
  --output-dir .
```

---

## 📊 Evaluation

Use the provided evaluation script:

```bash id="eval"
python eval.py -p dev1.out -g data/dev
python eval.py -p dev2.out -g data/dev
```
Ensure the paths for 'eval.py', 'dev1.out', 'dev2.out' and 'dev' data are correct.

---

## 🧠 Model Description

### Task 1

* Embedding → BiLSTM → Linear → ELU → Classifier
* Embedding dim: 100
* Hidden dim: 256
* Dropout: 0.33
* Optimizer: SGD

---

### Task 2

* Initialized embeddings using GloVe
* Added capitalization features

**Capitalization Strategy:**
To compensate for case-insensitive GloVe embeddings, capitalization features were incorporated (e.g., uppercase, title case), improving entity recognition.

---

### Bonus

* Character embeddings + CNN
* Combined with word embeddings
* Helps with rare and unseen words

---

## 🔁 Reproducibility

* Random seed fixed to **42**
* Deterministic setup used where applicable

---

## 📌 Output Format

All prediction files follow the required format:

```id="format"
index word predicted_tag
```

Sentences are separated by a blank line.

---

## ✅ Summary

| Task   | Output Files                   |
| ------ | ------------------------------ |
| Task 1 | blstm1.pt, dev1.out, test1.out |
| Task 2 | blstm2.pt, dev2.out, test2.out |
| Bonus  | bonus.pt, pred                 |

---

## ✅ End of README
