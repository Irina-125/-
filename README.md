[README.md](https://github.com/user-attachments/files/27105839/README.md)
# -# DialectAI v5 — English Dialect Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c)](https://pytorch.org)
[![HuBERT Large](https://img.shields.io/badge/HuBERT-Large%20LS960-orange)](https://huggingface.co/facebook/hubert-large-ls960-ft)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Automatic classification of **13 English regional dialects** from a 5-second audio clip using HuBERT Large + LightweightTDNN with ArcFace metric learning.

---

## Results (Validation set, 4 936 samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | **95.60%** |
| **UA (Macro Recall)** ★ | **91.05%** |
| Macro F1 | 89.93% |
| ROC-AUC (macro OvR) | 99.20% |
| Cohen κ | 0.9522 |
| Log-loss | 0.511 |
| Top-3 Accuracy | 99.53% |

★ — primary metric under class imbalance

### Per-class F1

| Dialect | F1 | Support |
|---------|----|---------|
| northern (England) | 1.000 | 455 |
| dutch (EFL) | 1.000 | 405 |
| american | 0.996 | 392 |
| scottish | 0.994 | 391 |
| indian | 0.994 | 397 |
| canadian | 0.991 | 388 |
| spanish (EFL) | 0.988 | 380 |
| irish | 0.988 | 412 |
| german (EFL) | 0.987 | 390 |
| czech (EFL) | 0.978 | 391 |
| polish (EFL) | 0.966 | 386 |
| **english (RP)** | **0.810** | 396 |
| england* | 0.000 | 153 |

*`england` overlaps with `english` — see Known Issues.

---

## Supported Dialects

| Dialect | Group | Notes |
|---------|-------|-------|
| `english` | British Isles | **Reference dialect (RP)** |
| `northern` | British Isles | Northern English accents |
| `scottish` | British Isles | Scottish English |
| `irish` | British Isles | Irish English |
| `american` | Americas | General American |
| `canadian` | Americas | Canadian English |
| `indian` | Asia | Indian English |
| `german` | Europe | German EFL accent |
| `dutch` | Europe | Dutch EFL accent |
| `czech` | Europe | Czech EFL accent |
| `polish` | Europe | Polish EFL accent |
| `spanish` | Europe | Spanish EFL accent |

---

## Architecture

```
Audio (5s, 16kHz)
    │
    ▼
HuBERT Large (frozen lower layers, 3/24 trainable)
    │  [B, T, 1024]
    ▼
SpeakerNormalizer (Instance Norm per channel)
    │
    ▼
LightweightTDNN (3 × TDNN layers, 256-dim)
    │
    ▼
Weighted Aggregation (attention pooling)
    │  [B, 256]
    ▼
L2-normalized embedding
    │
    ├─► ArcFace classifier → region probabilities
    └─► (Triplet loss during training)
```

**Training losses:** `L = ArcFace + 0.3 × TripletLoss`

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-repo/dialect-ai-v5.git
cd dialect-ai-v5

# 2. Install
pip install -r requirements.txt

# 3. Run inference
python predict.py --audio path/to/speech.wav

# 4. Run full notebook
jupyter notebook dialect_v5_enhanced.ipynb
```

### Single file inference

```python
from predict import Predictor
import pickle

with open("runs/dialect_v5/region_enc.pkl", "rb") as f:
    region_enc = pickle.load(f)

predictor = Predictor(model, region_enc)
result = predictor.predict("audio.wav")
print(result)
# {
#   "top_region": "american",
#   "region_probs": {"american": 0.921, "canadian": 0.065, ...},
#   "conformity": 0.81,          # distance from British RP
#   "conformity_tag": "Близко к эталону"
# }
```

---

## Dataset Structure

```
AccentDataset/
├── american_accent/
│   ├── american_accent_001.wav
│   └── ...
├── british_accent/          # → english
│   └── ...
├── indian_accent/
│   └── ...
└── ...
```

File naming: `<dialect_key><number>.wav`  
Total: **32 907 samples**, 13 classes, 5 s / 16 kHz mono WAV

---

## Training

```bash
# With HuBERT cache (recommended, ~10× faster after first run)
python train.py --use_cache True --epochs 50

# Without cache
python train.py --use_cache False --epochs 50
```

Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Weight decay | 1e-4 |
| ArcFace margin | 0.5 |
| ArcFace scale | 64 |
| Triplet margin | 0.3 |
| Batch size | 16 |
| Early stopping | 5 epochs |
| Trainable params | 906 241 |

---

## Conformity Analysis (Distance from British RP)

British and Irish dialects cluster closest to the RP reference in embedding space:

```
english  (RP)  ████████████████████  0.000  [REFERENCE]
england        ███████████████████▌  0.003  ≈ identical
northern       ██████████████████    0.120
scottish       █████████████████▌    0.150
irish          █████████████████     0.180
───────────────────────────────────────────
indian         ■                     1.100  EFL/non-native
german         ■                     1.050
american       ■                     1.197  Different native variety
canadian       ■                     1.267  Farthest from RP
```

---

## Project Structure

```
dialect-ai-v5/
├── dialect_v5_enhanced.ipynb   # Main notebook
├── predict.py                  # Inference script
├── train.py                    # Training script (extracted)
├── requirements.txt
├── README.md
├── LICENSE
└── runs/
    └── dialect_v5/
        ├── best_model.pt
        ├── region_enc.pkl
        ├── confusion_val.png
        └── ...
```

---

## Known Issues

- **`england` class (F1=0.000):** Label overlaps completely with `english` (RP).
  Both are phonetically identical in the dataset. Recommendation: merge or remove.
- **Inference on atypical speakers:** Model may misclassify individual files
  that differ significantly from the training distribution (speaker-dependent bias).

---

## Comparison with v4

| | v4 (DialectModel) | v5 (LightDialect) |
|--|--|--|
| **Task** | Region + Profession | Region only |
| **Architecture** | PhoneticTransformer (4L×1024) | LightweightTDNN (3L×256) |
| **Accuracy** | ~84.7% | **95.6%** |
| **Macro F1** | ~82.1% | **89.9%** |
| **ROC-AUC** | ~95.2% | **99.2%** |
| **Trainable params** | ~8.3M | **906K** |
| **Inference speed** | 1× | **~4× faster** |
| **HuBERT cache** | No | Yes |

---

## Authors

Подгорных И.В., Кованцев А.Н.

## References

1. HuBERT: Hsu et al., 2021 — [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
2. ArcFace: Deng et al., 2019 — [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
3. ECAPA-TDNN: Desplanques et al., 2020 — [arXiv:2005.07143](https://arxiv.org/abs/2005.07143)
