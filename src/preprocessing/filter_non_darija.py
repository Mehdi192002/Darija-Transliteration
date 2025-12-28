import pandas as pd
import google.generativeai as genai
import time
import re
import os
import socket

# --- CONFIGURATION ---
API_KEY = 'AIzaSyDVxg3meldtTVBtHkj_ldRzdJOY3JX0ntk' # Reusing key from generate_dataset.py
INPUT_FILE = os.path.join('..', '..', 'data', 'interim', 'darija_cleaned_dataset_robust.csv')
OUTPUT_FILE = os.path.join('..', '..', 'data', 'processed', 'darija_final_dataset.csv')

BATCH_SIZE = 50
DELAY_SECONDS = 4

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def wait_for_internet():
    if is_connected(): return
    print("\n   ⚠️  INTERNET LOST. Pausing...", end="", flush=True)
    while not is_connected():
        time.sleep(5)
    print("\n   ✅ Connection restored!")
    time.sleep(2)

def classify_batch(lines_list, batch_index):
    """
    Sends a batch to Gemini to classify as Darija or Not.
    Returns a list of booleans: True (Keep/Darija), False (Remove/Other).
    """
    formatted_input = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines_list)])
    
    prompt = f"""
    Role: Expert Language Classifier.
    Task: Identify which of the following sentences are **Moroccan Darija** (written in Latin/Arabizi or Arabic script).
    
    Instructions:
    - Return a numbered list corresponding exactly to the input.
    - For each sentence, write "YES" if it is Moroccan Darija.
    - Write "NO" if it is English, French, Spanish, Italian, Indonesian, Portuguese, pure Standard Arabic, or unintelligible gibberish.
    - Be strict. If it looks like a European language (e.g. "Is this for the 24th"), mark NO.
    - If it's mixed (Darija + French), mark YES.
    
    Input List:
    {formatted_input}
    
    Output Format (list of YES/NO):
    """
    
    while True:
        try:
            response = model.generate_content(prompt)
            text_response = response.text.strip()
            
            results = []
            for line in text_response.split('\n'):
                # clean "1. YES" -> "YES"
                clean = re.sub(r'^\d+\.?\s*', '', line).strip().upper()
                # Extract YES or NO
                if 'YES' in clean:
                    results.append(True)
                elif 'NO' in clean:
                    results.append(False)
                # If neither found immediately, might be empty or formatted weirdly.
                # But usually the model follows instructions.
            
            # Validation
            if len(results) == len(lines_list):
                return results
            
            # If mismatch, try to salvage or retry
            if len(results) > len(lines_list):
                 return results[:len(lines_list)]
            
            print(f"   [Batch {batch_index}] Mismatch: Sent {len(lines_list)}, Got {len(results)}. Retrying...")
            time.sleep(2)
                
        except Exception as e:
            print(f"   [Batch {batch_index}] Error: {e}")
            if not is_connected():
                wait_for_internet()
                continue
            time.sleep(10)

if __name__ == "__main__":
    print("1. Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Ensure columns exist, usually 'input_text' and 'target_text'
    # Based on view_file, columns are input_text, target_text
    
    all_texts = df['input_text'].astype(str).tolist()
    keep_mask = []
    
    total_rows = len(all_texts)
    total_batches = (total_rows // BATCH_SIZE) + 1
    
    print(f"   Found {total_rows} rows. Filtering...")
    
    for i in range(0, total_rows, BATCH_SIZE):
        batch = all_texts[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"   Processing Batch {batch_num}/{total_batches}...")
        
        batch_results = classify_batch(batch, batch_num)
        keep_mask.extend(batch_results)
        
        time.sleep(DELAY_SECONDS)

    # Apply mask
    # Handle ensure length matches if something went weird (though logic tries to enforce it)
    if len(keep_mask) != len(df):
        print(f"Warning: Mask length {len(keep_mask)} != DF length {len(df)}. Truncating/Padding.")
        keep_mask = keep_mask[:len(df)] + [False]*(len(df)-len(keep_mask))

    df_filtered = df[keep_mask]
    
    print(f"2. Filtering complete.")
    print(f"   Original count: {len(df)}")
    print(f"   Filtered count: {len(df_filtered)}")
    print(f"   Removed: {len(df) - len(df_filtered)} rows")
    
    df_filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"3. Saved to {OUTPUT_FILE}")
