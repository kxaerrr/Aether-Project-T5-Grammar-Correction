# Aether Project — T5 Grammar Correction

Aether Project is a comprehensive grammar-correction system powered by the T5-small Transformer model. It performs end-to-end training, randomized hyperparameter tuning, evaluation with BLEU/ROUGE metrics, and final retraining based on optimal configurations. This project is structured to resemble modern Transformer-based research repositories, with clear documentation, reproducible experiments, and detailed logging.

---

## 🌟 Executive Summary

Aether Project automates the full lifecycle of Transformer training for grammar correction:

- Loads the **BEA-19 corruption dataset**
- Tokenizes text for seq2seq grammar transformation
- Runs **random-search hyperparameter tuning**
- Automatically identifies optimal hyperparameters
- Retrains the model using best parameters
- Saves the final model to `best_model/`
- Computes evaluation metrics using **BLEU and ROUGE**
- Provides an inference module for testing grammar correction
- Provides an interface that can be seen in https://f055857182469bd6e9.gradio.live/

This makes the project an excellent template for sequence-to-sequence NLP tasks and experiment management.

---

# 📁 Repository Structure

```
Aether-Project-T5-Grammar-Correction/
│
├── aether_project.py                         # Main training & tuning pipeline
├── best_model/                               # Stored best model (tracked via Git LFS)
├── aether_hyperparameter_tuning_log.xlsx     # Experiment log with metrics
├── requirements.txt                           # Python dependencies
├── README.md                                  # Full project documentation
└── .gitignore                                 # Ignored files
```

---

# 🔧 Technical Specifications

## Model Summary

| Component | Description |
|----------|-------------|
| **Model Type** | T5-small (Text-to-Text Transformer) |
| **Task** | Grammar Correction (Seq2Seq) |
| **Dataset** | BEA-19 Corruption Dataset |
| **Tuning Method** | Random Search (10 trials) |
| **Metrics** | BLEU, ROUGE-1, ROUGE-L, Eval Loss |
| **Tokenizer** | SentencePiece-based T5 tokenizer |
| **Framework** | Hugging Face Transformers |

---

# ⚙️ Hyperparameters Used (Random Search Space)

| Hyperparameter | Range / Values Tested |
|----------------|------------------------|
| Learning Rate | 1e-5 → 1e-4 |
| Batch Size | 1, 2, 4, 8 |
| Epochs | 2, 3, 4 |
| Gradient Accumulation | 1, 2, 4, 8 |
| Max Grad Norm | 0.5 → 2.0 |
| Adam Epsilon | 1e-8 → 1e-7 |
| Warmup Steps | 0, 300, 600 |
| Weight Decay | 0 → 0.1 |
| Label Smoothing | 0.0 → 0.1 |

The script automatically logs BLEU, ROUGE, loss, and training time for each configuration.

---

# 📊 Example: Hyperparameter Tuning Log (Preview)

| Trial | LR | Batch | Epochs | Accum | Loss | BLEU | ROUGE-L | Time (s) |
|------|----|-------|--------|--------|------|------|-----------|-----------|
| 1 | 2.1e-5 | 4 | 3 | 2 | 1.942 | 47.2 | 51.0 | 238 |
| 2 | 8.9e-5 | 8 | 2 | 1 | 2.304 | 39.7 | 43.9 | 211 |
| 3 | 4.5e-5 | 2 | 4 | 4 | 1.783 | 49.5 | 53.2 | 298 |
| … | … | … | … | … | … | … | … | … |

Actual detailed logs are stored in:

```
aether_hyperparameter_tuning_log.xlsx
```

---

# 🚀 Quick Start Guide

## 1. Clone the Repository
```
git clone https://github.com/kxaerrr/Aether-Project-T5-Grammar-Correction.git
cd Aether-Project-T5-Grammar-Correction
```

## 2. Install Dependencies
```
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## 3. Run Training & Tuning
```
python aether_project.py
```

This will:
- Run random search tuning
- Log all metrics
- Save the best model
- Retrain the model using best parameters
- Run evaluation and inference samples

---

# 📚 Dataset Information

This project uses:

```
Dataset: juancavallotti/bea-19-corruption
```

Loaded via:
```python
from datasets import load_dataset
dataset = load_dataset("juancavallotti/bea-19-corruption")
```

Training subset:
```
0 → 5000
```

Evaluation subset:
```
3000 → 3500
```

---

# 🧪 Inference Demo

After training completes, use:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("./best_model")
tokenizer = T5Tokenizer.from_pretrained("./best_model")

text = "grammar: She are going to the store."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Example Output:
```
She is going to the store.
```

---

# 📈 Evaluation & Metrics

Metrics used:

| Metric | Description |
|--------|-------------|
| **BLEU** | Measures n-gram overlap between predicted and corrected sentences |
| **ROUGE-1** | Measures recall of unigrams |
| **ROUGE-L** | Longest Common Subsequence score |
| **Eval Loss** | Cross-entropy loss during validation |

BLEU and ROUGE computed using:
```python
evaluate.load("bleu")
evaluate.load("rouge")
```

---

# 💾 Model Saving

The best model is automatically saved into:

```
./best_model
```

It includes:
- `pytorch_model.bin` or `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

---

# 🧲 Git LFS Instructions (Important)

GitHub rejects files > 100MB.

Enable LFS:
```
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
git add .
git commit -m "Track model files with LFS"
git push
```

If your repo is already contaminated with large files:
```
rm -rf .git
git init
git lfs install
git add .
git commit -m "Clean repo and reinitialize with LFS"
git remote add origin https://github.com/kxaerrr/Aether-Project-T5-Grammar-Correction.git
git push --force
```

---

# 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not detected | Ensure CUDA is installed / Use Google Colab |
| Out of memory | Lower batch size, reduce dataset size, reduce max_length |
| Large file push error | Use Git LFS and clean repo history |
| Training too slow | Use GPU + reduce dataset size |
| Poor accuracy | Increase number of tuning trials |

---

# 📝 License

This project is provided for **research and educational** purposes. You may modify, extend, and share with attribution.

---

# 🤝 Contributing

Contributions are welcome. You may:
- Add more datasets  
- Add wandb logging  
- Improve tuning strategies  
- Upgrade to T5-base or FLAN-T5  
- Refactor into multi-file project  

Open a PR anytime.

---

# 🎯 Final Notes

Aether Project is intentionally designed as both:
1. A production-ready grammar-correction benchmark framework  
2. A learning-friendly example of full-cycle NLP model training  

It is flexible, modular, and easily extensible.

Enjoy experimenting with Transformer models!
