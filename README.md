# Aether Project — T5 Grammar Correction

Aether Project is a T5-based grammar correction system that trains, tunes, evaluates, and retrains a grammar model using BLEU and ROUGE metrics on the BEA-19 corruption dataset. It performs random-search hyperparameter tuning, automatically identifies the best-performing configuration, and saves a fully retrained final model for inference.

Aether Project fine-tunes the t5-small model for grammar correction using custom preprocessing, sentence-based input/label mapping, and metric-based model selection. Training includes random hyperparameter exploration, evaluation after each trial, and final model refinement using the best parameters discovered through tuning. All results are logged in an Excel file for transparency and reproducibility, and the best model is stored inside the `best_model/` directory for further use.

The full repository includes the main script (`aether_project.py`), which handles dataset loading, tokenization, training, evaluation, logging, metric computation, best-model saving, final retraining, and inference demonstration. The script is ready for execution in both local environments and Google Colab, and supports GPU acceleration automatically. A custom test dataset is included inside the script to demonstrate real-world grammar correction after training is completed.

To get started, clone the repository using:
```
git clone https://github.com/kxaerrr/Aether-Project-T5-Grammar-Correction.git
cd Aether-Project-T5-Grammar-Correction
```
and run the full script using:
```
python aether_project.py
```

You must have Python 3.8+, Git LFS for large model files, and a GPU for efficient training (optional but recommended). Install dependencies using:
```
pip install -r requirements.txt
python -m nltk.downloader punkt
```
with a recommended requirements.txt:
```
transformers
datasets
evaluate
rouge_score
nltk
torch
numpy
pandas
scipy
openpyxl
```

The dataset used in this project is:
```
juancavallotti/bea-19-corruption
```
loaded via:
```python
from datasets import load_dataset
dataset = load_dataset("juancavallotti/bea-19-corruption")
```
The script automatically selects training and evaluation subsets to reduce memory usage:
```
train_data = dataset["train"].select(range(5000))
eval_data = dataset["train"].select(range(3000, 3500))
```

Hyperparameter tuning is performed using a random search loop inside the script, testing 10 random configurations by default. Each configuration trains T5-small with different values for learning rate, batch size, epochs, gradient accumulation, epsilon, warmup steps, label smoothing, and weight decay. The script measures evaluation loss, computes BLEU and ROUGE, and logs the results into `aether_hyperparameter_tuning_log.xlsx`. The model is saved whenever a new best eval loss is found.

After tuning completes, the script reads the Excel log, extracts the best hyperparameters, rebuilds the TrainingArguments with these optimal settings, retrains the model, and saves the final version again to:
```
./best_model
```

Inference is demonstrated with a small custom dataset included in the script. The best model and tokenizer are loaded from:
```python
model = T5ForConditionalGeneration.from_pretrained("./best_model")
tokenizer = T5Tokenizer.from_pretrained("./best_model")
```
You can then test grammar correction like:
```python
text = "grammar: I like play soccer."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

BLEU and ROUGE metrics are computed using the `evaluate` library:
```python
evaluate.load("bleu")
evaluate.load("rouge")
```
These metrics are used to validate tuning and measure performance of the final model.

Because model files are often larger than 100MB, Git LFS is required to push the model to GitHub. Enable LFS using:
```
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
git add .
git commit -m "Add model with LFS"
git push
```

If large files were already committed before enabling LFS, delete your `.git` folder and reinitialize:
```
rm -rf .git
git init
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .
git commit -m "Reinitialize with LFS"
git remote add origin https://github.com/kxaerrr/Aether-Project-T5-Grammar-Correction.git
git push --force origin main
```

Troubleshooting notes: if your system runs out of memory, reduce batch size or dataset size, or use gradient accumulation. If model pushes fail, ensure LFS is tracking your large files. If training is slow, use a GPU. If you want deterministic results, set seeds within the script.

The repository and model are provided for educational and research purposes. Contributions are welcome, including improvements to the training script, modularization, expanded dataset support, and improved evaluation tools.
