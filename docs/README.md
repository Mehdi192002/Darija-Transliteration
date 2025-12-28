# Darija Transliteration System

A comprehensive machine learning pipeline for bidirectional transliteration between Moroccan Darija (Arabic script) and Latin script (Arabizi), built using transformer-based models.

## 📋 Overview

This project provides a complete end-to-end solution for:
- **Dataset Generation**: Converting Arabizi to Arabic script using LLM-powered transliteration
- **Model Training**: Fine-tuning ByT5 models for accurate bidirectional transliteration
- **Inference**: Real-time transliteration with intelligent word-level processing

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas transformers datasets torch scikit-learn google-generativeai
```

### 1. clean Dataset (Optional - datasets provided)

```bash
# Refine and filter the dataset
python clean_and_refine.py
python filter_non_darija.py
```

### 2. Train the Model

```bash
# Stage 1: Initial training
python train_model.py

# Stage 2: Generate synthetic data
python generate_fake_words.py

# Stage 3: Fine-tune with augmented data
python finetune_model.py
```

### 3. Use the Model

```bash
# Interactive transliteration
python use_model.py
```

**Example Usage:**
```
Arabic Input: كيف داير خويا؟
Latin Output: kif dayer khoya?

Arabic Input: واش بغيتي تمشي معايا؟
Latin Output: wach bghiti tmchi m3aya?
```

## 📊 Dataset Statistics

| Dataset | Rows | Description |
|---------|------|-------------|
| Raw Comments | ~1,000 | Social media comments (Instagram Reels) |
| Cleaned Robust | 863 | Validated Darija sentences |
| Final Dataset | 808 | Quality-filtered pairs |
| Word Pairs | ~2,000 | Word-level alignments |
| Synthetic | 5,000 | Rule-based generated pairs |
| Augmented | ~6,000 | Combined training set |

## 🔧 Configuration

### API Keys
The project uses Google's Gemini API for dataset generation. Add your API key in:
- `generate_dataset.py`
- `clean_and_refine.py`
- `pair_words.py`

```python
API_KEY = 'YOUR_API_KEY_HERE'
```

### Model Selection
- **Base Model**: `google/byt5-small` (byte-level T5)
- **LLM Models**: `gemini-2.0-flash`, `gemma-3-27b-it`, `gemma-3-12b-it`

### Training Parameters
```python
EPOCHS = 5              # Initial training
BATCH_SIZE = 16         # Adjust based on GPU memory
LEARNING_RATE = 1e-4    # Initial training
FINE_TUNE_LR = 4e-5     # Fine-tuning (lower for stability)
```

## 🎓 Model Architecture

The system uses **ByT5** (Byte-level T5), which:
- Operates on raw bytes instead of subword tokens
- Handles multilingual text naturally (Arabic + Latin)
- Requires no language-specific preprocessing
- Excels at character-level transformations

**Training Strategy:**
1. **Base Training**: Learn from real-world sentence pairs
2. **Synthetic Augmentation**: Reinforce specific character mappings (3→ع, 9→ق, etc.)
3. **Fine-tuning**: Combine both datasets for robust performance

## 📝 Special Character Mappings

The model learns these common Arabizi conventions:

| Latin | Arabic | Example |
|-------|--------|---------|
| 3 | ع | m3a → معا |
| 9 | ق | 9alb → قلب |
| 7 | ح | 7al → حال |
| 2 | أ | 2ana → أنا |
| 5 | خ | 5oya → خويا |
| 4 | غ | 4ir → غير |
| x | ش | xkoun → شكون |

## 🔍 Data Processing Pipeline

```
Raw Data (Instagram Comments)
    ↓
[generate_dataset.py] → Gemini API transliteration
    ↓
[clean_and_refine.py] → Quality filtering, number conversion
    ↓
[filter_non_darija.py] → Language detection
    ↓
[pair_words.py] → Word-level extraction
    ↓
Final Training Dataset
```

## 🧪 Evaluation

Use `test_model.py` to evaluate model performance on held-out test sets.

**Metrics:**
- Character Error Rate (CER)
- Word Error Rate (WER)
- BLEU Score

## 🛠️ Advanced Features

### Smart Paragraph Processing (`use_model.py`)
- **Number Preservation**: Keeps digits unchanged (e.g., "2024" → "2024")
- **Single Character Handling**: Attaches prefixes like "و" to next word
- **Punctuation Awareness**: Maintains sentence structure

### Robust Error Handling
- Network interruption recovery
- API quota management with model switching
- Resume-from-checkpoint support

### Multi-Model Fallback
The system automatically switches between models if quota limits are hit:
1. `gemma-3-27b-it`
2. `gemini-2.0-flash-exp`
3. `gemini-2.0-flash`
4. `gemini-flash-latest`

## 📈 Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for 10x faster training
   ```bash
   python clean_pair.py  # Check GPU availability
   ```

2. **Batch Size Tuning**: Increase batch size if you have more VRAM
   ```python
   BATCH_SIZE = 32  # For GPUs with 16GB+ VRAM
   ```

3. **Mixed Precision**: Enable FP16 on Linux/CUDA for faster training
   ```python
   fp16=True  # In training arguments (Windows users: keep False)
   ```

**Note**: This project is part of a Master's program in Data Science and Analytics (2024-2025) - Natural Language Processing course.
