# Quick Reference Guide

## 🚀 Common Commands

### Training
```bash
# Full training pipeline
python train_model.py              # ~30 min on GPU
python generate_fake_words.py      # ~1 second
python finetune_model.py           # ~15 min on GPU
```

### Inference
```bash
# Interactive mode
python use_model.py

# Test mode
python test_model.py
```

### Data Processing
```bash
# Generate new dataset
python generate_dataset.py         # Requires API key, ~1 hour

# Clean and filter
python clean_and_refine.py         # Requires API key, ~30 min
python filter_non_darija.py        # Requires API key, ~20 min

# Extract word pairs
python pair_words.py               # Requires API key, ~15 min
```

## 📊 File Sizes

| File | Size | Description |
|------|------|-------------|
| moroccan_corpus.jsonl | 711 MB | Large corpus (optional) |
| darija_final_dataset.csv | 154 KB | Main training data |
| darija_word_pairs.csv | 136 KB | Word-level pairs |
| synthetic_dataset.csv | 78 KB | Generated examples |
| Model directories | ~500 MB each | Trained models |

## 🔑 Character Mappings (Arabizi → Arabic)

```
Numbers:
2 → أ    3 → ع    4 → غ    5 → خ
7 → ح    9 → ق

Letters:
a → ا    b → ب    d → د    f → ف
h → ه    j → ج    k → ك    l → ل
m → م    n → ن    r → ر    s → س
t → ت    w → و    x → ش    y → ي
z → ز
```

## 📝 Example Translations

| Latin (Arabizi) | Arabic Script |
|-----------------|---------------|
| salam | سلام |
| labas | لباس |
| kif dayer | كيف داير |
| wach bghiti | واش بغيتي |
| m3a | معا |
| 3la | على |
| 9alb | قلب |
| 7al | حال |
| khoya | خويا |
| bezzaf | بزاف |

## ⚙️ Configuration Quick Reference

### API Keys Location
- `generate_dataset.py` line 9
- `clean_and_refine.py` line 10
- `pair_words.py` line 9

### Model Paths
```python
# Base model
MODEL_PATH = "./darija_transliteration_model"

# Fine-tuned model (recommended)
MODEL_PATH = "./darija_transliteration_model_v2"
```

### Training Parameters
```python
# train_model.py
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# finetune_model.py
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 4e-5  # Lower for fine-tuning
```

### Dataset Generation
```python
# generate_dataset.py
BATCH_SIZE = 100        # Sentences per API call
DELAY_SECONDS = 30      # Wait between calls

# generate_dataset_bach.py
BATCH_SIZE = 15         # Smaller batches
DELAY_SECONDS = 15      # Faster processing
```

## 🐛 Common Errors & Fixes

### Error: "No module named 'transformers'"
```bash
pip install transformers datasets
```

### Error: "CUDA out of memory"
```python
# Reduce batch size in training script
BATCH_SIZE = 8  # or 4
```

### Error: "API quota exceeded"
```python
# Increase delay in generation scripts
DELAY_SECONDS = 60  # Wait longer between calls
```

### Error: "Model not found"
```bash
# Check if model exists
dir darija_transliteration_model_v2  # Windows
ls darija_transliteration_model_v2   # Linux/Mac

# If missing, train the model first
python train_model.py
```

## 📈 Performance Optimization

### Speed up training
1. Use GPU (10x faster)
2. Increase batch size (if you have VRAM)
3. Use mixed precision (Linux/CUDA only)

### Improve accuracy
1. Add more diverse training data
2. Fine-tune with domain-specific examples
3. Increase training epochs
4. Use larger model (ByT5-base instead of small)

### Reduce memory usage
1. Decrease batch size
2. Use gradient accumulation
3. Clear cache between batches

## 🔄 Workflow Diagrams

### Training Workflow
```
Data → Clean → Train → Generate Synthetic → Fine-tune → Deploy
 ↓       ↓       ↓            ↓                ↓          ↓
CSV    Filter  Model    synthetic.csv      Model v2    use_model.py
```

### Inference Workflow
```
User Input → Tokenize → Model → Decode → Output
   ↓            ↓         ↓        ↓        ↓
"سلام"      [IDs]    Transform  [IDs]   "salam"
```

## 💡 Tips & Tricks

1. **Always use v2 model**: It's fine-tuned and more accurate
2. **GPU is essential**: Training on CPU takes hours
3. **Validate outputs**: AI can make mistakes, review results
4. **Backup models**: Save trained models externally
5. **Version control**: Use git for code, not for large files
6. **API limits**: Be mindful of quota (5 req/min for free tier)
7. **Data quality**: Better data = better model

## 📞 Quick Help

- **Installation issues**: See SETUP.md
- **Usage questions**: See README.md
- **Want to contribute**: See CONTRIBUTING.md
- **Found a bug**: Open GitHub issue
- **Need examples**: Check the CSV files

## 🎯 Project Goals

- ✅ Bidirectional transliteration (Arabic ↔ Latin)
- ✅ Word and sentence level support
- ✅ Robust to typos and variations
- ✅ Fast inference (<100ms per sentence)
- ⏳ Web interface (coming soon)
- ⏳ Mobile app (future)

---

**Last Updated**: December 2024
**Version**: 2.0
