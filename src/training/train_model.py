import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import torch
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = os.path.join('..', '..', 'data', 'processed', 'darija_dataset_clean_final.csv')
MODEL_NAME = "google/byt5-small"
OUTPUT_DIR = os.path.join("..", "..", "models", "base")
EPOCHS = 5  # Increased slightly to ensure convergence
BATCH_SIZE = 16

def prepare_data():
    print("1. Loading Data...")
    try:
        # Load CSV (Col 0 = Latin, Col 1 = Arabic)
        df = pd.read_csv(INPUT_FILE, header=None, names=['latin', 'arabic'])
        # Drop any failed rows
        df = df.dropna().astype(str)
        
        # BASIC VALIDATION: Check if data looks swapped
        # Arabic usually has characters > \u0600. Latin < \u00ff
        first_arabic = df['arabic'].iloc[0]
        if not any('\u0600' <= c <= '\u06FF' for c in first_arabic):
            print("⚠️ WARNING: Your columns might be swapped! Checking...")
            # If Col 1 isn't Arabic, assume Col 0 is Arabic?
            # For now, we trust your previous steps, but keep an eye on this.
        
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        print(f"   Training Examples: {len(train_dataset)}")
        print(f"   Testing Examples:  {len(test_dataset)}")
        
        return train_dataset, test_dataset
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit()

def main():
    # 1. Prepare Data
    train_data, test_data = prepare_data()
    
    # 2. Load Tokenizer & Model
    print(f"2. Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 3. Preprocessing Function (FIXED)
    def preprocess_function(examples):
        inputs = examples["arabic"]
        targets = examples["latin"]
        
        # Tokenize inputs (Dynamic padding handled by collator later)
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing
    tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
    tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

    # 4. Data Collator (CRITICAL FIX)
    # This automatically pads inputs to the longest in the batch
    # AND replaces padding tokens in labels with -100 so the model ignores them
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        label_pad_token_id=-100 # This ensures we don't learn to predict "empty space"
    )

    # 5. Training Setup
    args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-4, 
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            save_total_limit=1,
            num_train_epochs=EPOCHS,
            predict_with_generate=True,
            fp16=False,                 # <--- CRITICAL FIX: Set this to False
            logging_steps=50,
            load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train!
    print("3. Starting Training...")
    trainer.train()

    # 7. Save Final Model
    print(f"4. Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! Model is ready.")

if __name__ == "__main__":
    main()