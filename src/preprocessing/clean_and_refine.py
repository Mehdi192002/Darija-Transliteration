import pandas as pd
import google.generativeai as genai
import time
import re
import os
import socket
import json

# --- CONFIGURATION ---
API_KEY = 'AIzaSyDVxg3meldtTVBtHkj_ldRzdJOY3JX0ntk' # Reusing key
INPUT_FILE = os.path.join('..', '..', 'data', 'interim', 'darija_cleaned_dataset_robust.csv')
OUTPUT_FILE = os.path.join('..', '..', 'data', 'processed', 'darija_final_dataset.csv')

BATCH_SIZE = 5 # Very small batch to avoid token limits
DELAY_SECONDS = 15 # Very generous delay to stay under RPM

# Global list of models to try
MODELS = ['gemma-3-27b-it', 'gemini-2.0-flash-exp', 'gemini-2.0-flash', 'gemini-flash-latest']
current_model_idx = 0

genai.configure(api_key=API_KEY)

def get_current_model():
    global current_model_idx
    return genai.GenerativeModel(MODELS[current_model_idx])

def switch_model():
    global current_model_idx
    current_model_idx = (current_model_idx + 1) % len(MODELS)
    print(f"   ⚠️  Switching to model: {MODELS[current_model_idx]}")

def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def wait_for_network():
    if is_connected(): return
    print("\n   ⚠️  Network lost. Waiting...", end="", flush=True)
    while not is_connected():
        time.sleep(5)
    print("\n   ✅ Resumed.")
    time.sleep(2)

def process_batch(rows):
    """
    rows: list of dicts [{'original': ..., 'current_arabic': ...}]
    Returns list of dicts [{'refined_arabic': ..., 'is_darija': ...}]
    """
    
    # Construct prompt
    lines_text = []
    for i, row in enumerate(rows):
        lines_text.append(f"ITEM {i+1}:")
        lines_text.append(f"  Arabizi: {row['original']}")
        lines_text.append(f"  Current Arabic: {row['current_arabic']}")
    
    input_block = "\n".join(lines_text)
    
    prompt = f"""
    Role: Expert Moroccan Darija Linguist.
    
    Task: Review and process {len(rows)} items.
    For each item:
    1. **Check if it is Darija**. If it's mostly English, French, Spanish, or Standard Arabic, mark IS_DARIJA as False. If it's mixed or Darija, mark True.
    2. **Refine the Arabic Script**:
       - The 'Current Arabic' might have errors like "3" instead of "ع", "9" instead of "ق", "7" instead of "ح". FIX THESE. (e.g. "بو3و" -> "بوعو").
       - The User wants **NO NUMBERS** (digits) in the Arabic text. Convert all digits to their Darija word equivalents. (e.g. "4" -> "ربعة", "100" -> "مية", "2024" -> "ألفين وأربعة وعشرين"). ALWAYS spell out numbers.
       - Ensure the Arabic matches the meaning of the Arabizi.
    
    Input:
    {input_block}
    
    Output Format:
    Return a valid JSON array of objects.
    Example:
    [
      {{"id": 1, "refined_arabic": "سلام خويا، لباس؟", "is_darija": true}},
      {{"id": 2, "refined_arabic": "", "is_darija": false}}
    ]
    
    Strict JSON only. No markdown formatting.
    """
    
    retries = 0
    # Try somewhat infinitely, cycling models
    while retries < 10:
        try:
            model = get_current_model()
            response = model.generate_content(prompt)
            # Clean response
            txt = response.text.replace('```json', '').replace('```', '').strip()
            data = json.loads(txt)
            
            # Validation
            if len(data) != len(rows):
                print(f"Warning: Count mismatch ({len(data)} vs {len(rows)}). Retrying...")
                retries += 1
                continue
                
            return data
            
        except Exception as e:
            err = str(e)
            print(f"   Batch Error: {err[:100]}...")
            
            if "429" in err or "quota" in err.lower():
                print("   Quota hit. Switching model...")
                switch_model()
                time.sleep(5)
            elif not is_connected():
                wait_for_network()
            elif "404" in err: # Model not found
                print("   Model not found. Switching...")
                switch_model()
                time.sleep(2)
            else:
                retries += 1
                time.sleep(5)
    
    # If failed after many retries
    return [{"refined_arabic": "ERROR", "is_darija": False}] * len(rows)

if __name__ == "__main__":
    print("1. Reading input...")
    df = pd.read_csv(INPUT_FILE)
    # Ensure columns. Typically 0=input, 1=target or named 'input_text', 'target_text'
    if 'input_text' not in df.columns:
        # Fallback if headerless
        df.columns = ['input_text', 'target_text']
    
    print(f"   Count: {len(df)}")
    
    # Check for existing work
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            done_df = pd.read_csv(OUTPUT_FILE)
            start_idx = len(done_df)
            print(f"   Resuming from row {start_idx}...")
        except:
             # Create empty
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("input_text,target_text\n")

    else:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("input_text,target_text\n")
            
    # Processing
    all_data = []
    
    # We iterate from start_idx
    inputs = df['input_text'].tolist()
    targets = df['target_text'].tolist()
    
    for i in range(start_idx, len(df), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(df))
        batch_inputs = inputs[i:batch_end]
        batch_targets = targets[i:batch_end]
        
        batch_rows = []
        for j in range(len(batch_inputs)):
            batch_rows.append({
                'original': str(batch_inputs[j]),
                'current_arabic': str(batch_targets[j])
            })
            
        print(f"   Processing {i} to {batch_end}...")
        
        results = process_batch(batch_rows)
        
        # Save results immediately
        save_rows = []
        for idx, res in enumerate(results):
            if res.get('is_darija', True) and res.get('refined_arabic') != "ERROR":
                # Clean input text of commas to avoid csv breakage (basic)
                inp = batch_rows[idx]['original'].replace('"', '""')
                out = res['refined_arabic'].replace('"', '""')
                save_rows.append(f'"{inp}","{out}"')
        
        if save_rows:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write("\n".join(save_rows) + "\n")
                
        time.sleep(DELAY_SECONDS)

    print("Done!")
