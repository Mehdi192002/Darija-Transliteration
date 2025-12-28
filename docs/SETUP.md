# Darija Transliteration - Setup Guide

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/darija-transliteration.git
cd darija-transliteration
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. GPU Setup (Optional but Recommended)

For CUDA-enabled GPU acceleration:

```bash
# Check if CUDA is available
python clean_pair.py

# Install PyTorch with CUDA support (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. API Configuration

Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

Update the following files with your API key:
- `generate_dataset.py` (line 9)
- `clean_and_refine.py` (line 10)
- `pair_words.py` (line 9)

```python
API_KEY = 'YOUR_API_KEY_HERE'
```

## Usage Workflows

### Workflow 1: Use Pre-trained Model (Quickest)

If you have the trained models already:

```bash
python use_model.py
```

### Workflow 2: Train from Scratch

```bash
# Step 1: Generate dataset (requires API key)
python generate_dataset.py

# Step 2: Clean and refine
python clean_and_refine.py

# Step 3: Filter non-Darija content
python filter_non_darija.py

# Step 4: Extract word pairs
python pair_words.py

# Step 5: Train base model
python train_model.py

# Step 6: Generate synthetic data
python generate_fake_words.py

# Step 7: Combine datasets
# Windows:
type darija_dataset_clean_final.csv synthetic_dataset.csv > darija_dataset_augmented.csv

# Linux/Mac:
cat darija_dataset_clean_final.csv synthetic_dataset.csv > darija_dataset_augmented.csv

# Step 8: Fine-tune model
python finetune_model.py

# Step 9: Use the model
python use_model.py
```

### Workflow 3: Fine-tune Existing Model

If you have a base model and want to improve it:

```bash
# Generate synthetic data
python generate_fake_words.py

# Combine with existing data
type darija_dataset_clean_final.csv synthetic_dataset.csv > darija_dataset_augmented.csv

# Fine-tune
python finetune_model.py

# Test
python use_model.py
```

## Troubleshooting

### Issue: "No module named 'transformers'"

```bash
pip install transformers datasets
```

### Issue: "CUDA out of memory"

Reduce batch size in training scripts:

```python
BATCH_SIZE = 8  # or even 4
```

### Issue: "API quota exceeded"

The scripts automatically switch between models. Wait a few minutes or:
- Use a different API key
- Reduce `BATCH_SIZE` in dataset generation scripts
- Increase `DELAY_SECONDS`

### Issue: "Model not found"

Make sure you've completed training:

```bash
# Check if model directories exist
dir darija_transliteration_model      # Windows
ls darija_transliteration_model       # Linux/Mac
```

### Issue: Training is very slow

- Enable GPU acceleration (see step 4 above)
- Reduce dataset size for testing
- Use a smaller model (ByT5-small is already the smallest)

## Directory Structure After Setup

```
DataProcessing/
├── venv/                                    # Virtual environment
├── darija_transliteration_model/           # Base model (after training)
├── darija_transliteration_model_v2/        # Fine-tuned model
├── *.csv                                    # Datasets
├── *.py                                     # Scripts
└── README.md
```

## Performance Benchmarks

**Training Time (on different hardware):**

| Hardware | Base Training | Fine-tuning | Total |
|----------|---------------|-------------|-------|
| CPU (8 cores) | ~2-3 hours | ~1 hour | ~3-4 hours |
| GPU (RTX 3060) | ~20 minutes | ~10 minutes | ~30 minutes |
| GPU (RTX 4090) | ~8 minutes | ~4 minutes | ~12 minutes |

**Inference Speed:**

| Hardware | Words/second |
|----------|--------------|
| CPU | ~5-10 |
| GPU | ~50-100 |

## Next Steps

1. **Test the model**: Run `python use_model.py` and try various inputs
2. **Evaluate performance**: Use `python test_model.py`
3. **Collect more data**: Add your own Darija examples to improve accuracy
4. **Fine-tune further**: The more diverse data you add, the better the model becomes

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review the code comments in each script
- Open an issue on GitHub for bugs or questions

## Tips for Best Results

1. **Data Quality**: The model is only as good as the training data
2. **Balanced Dataset**: Include diverse examples (formal, informal, mixed languages)
3. **Regular Fine-tuning**: Add new examples and fine-tune periodically
4. **GPU Usage**: Always use GPU for training if available
5. **Validation**: Manually review model outputs to catch errors

---

Happy transliterating! 🇲🇦
