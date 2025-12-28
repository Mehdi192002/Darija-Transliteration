import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import torch
import os

# --- CONFIGURATION ---
# 1. We load the NEW augmented dataset (Real + Synthetic)
INPUT_FILE = os.path.join('..', '..', 'data', 'processed', 'darija_dataset_augmented.csv')

# 2. We load YOUR existing trained model (not the empty one from Google)
MODEL_PATH = os.path.join("..", "..", "models", "base")

# 3. We save to a NEW folder (Version 2)
OUTPUT_DIR = os.path.join("..", "..", "models", "v2")

EPOCHS = 3           # We don't need many epochs for fine-tuning
BATCH_SIZE = 16      # Keep 16 for GPU

def prepare_data():
    print(f"1. Loading Augmented Data from {INPUT_FILE}...")
    try:
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Error: {INPUT_FILE} not found!")
            print("   Did you run: type darija_dataset_clean_final.csv synthetic_dataset.csv > darija_dataset_augmented.csv")
            exit()

        df = pd.read_csv(INPUT_FILE, header=None, names=['latin', 'arabic'])
        df = df.dropna().astype(str)
        
        # Split
        train_df, test_df = train_test_split(df, test_size=0.05, random_state=42)
        
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        print(f"   Training Examples: {len(train_dataset)}")
        print(f"   Testing Examples:  {len(test_dataset)}")
        
        return train_dataset, test_dataset
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

def main():
    # 1. Prepare Data
    train_data, test_data = prepare_data()
    
    # 2. Load YOUR Existing Model
    print(f"2. Loading previous model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    except OSError:
        print("❌ Error: Could not find your previous model folder.")
        print("   Make sure './darija_transliteration_model' exists.")
        exit()

    # 3. Preprocessing
    def preprocess_function(examples):
        inputs = examples["arabic"]
        targets = examples["latin"]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
    tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    # 4. Training Setup
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=4e-5,          # Lower rate for fine-tuning (gentle updates)
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=1,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=False,                  # Keep False for stability on Windows
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

    # 5. Train!
    print("3. Starting Fine-Tuning...")
    trainer.train()

    # 6. Save Version 2
    print(f"4. Saving NEW model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! Model V2 is ready.")

if __name__ == "__main__":
    main()