import pandas as pd
import google.generativeai as genai
import time
import re
import os
import socket

# --- CONFIGURATION ---
API_KEY = 'AIzaSyDVxg3meldtTVBtHkj_ldRzdJOY3JX0ntk'  # <--- PASTE YOUR KEY HERE
INPUT_FILE = os.path.join('..', '..', 'data', 'raw', 'darija_reels_comments.csv')
OUTPUT_FILE = os.path.join('..', '..', 'data', 'interim', 'darija_cleaned_dataset_robust.csv')

BATCH_SIZE = 100  
DELAY_SECONDS = 30 

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemma-3-27b-it')

def is_connected():
    """
    Simple check to see if we can reach Google's DNS servers.
    """
    try:
        # Try to connect to 8.8.8.8 (Google DNS) on port 53
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def wait_for_internet():
    """
    Loops indefinitely until internet connection is detected.
    """
    if is_connected():
        return

    print("\n   ⚠️  INTERNET LOST. Pausing script...", end="", flush=True)
    while not is_connected():
        time.sleep(5) # Check every 5 seconds
        print(".", end="", flush=True)
    
    print("\n   ✅ Connection restored! Resuming in 5 seconds...\n")
    time.sleep(5) # Let connection stabilize

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.replace("reel_comment", "")
    text = " ".join(text.split())
    if len(text) < 2: return ""
    return text

def process_mega_batch_robust(lines_list, batch_index):
    """
    Sends a batch to Gemini.
    Crucially: It loops FOREVER until it gets a success or a non-network error.
    """
    formatted_input = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines_list)])
    
    prompt = f"""
    Role: Expert Moroccan Darija Translator.
    Task: Transliterate the following {len(lines_list)} sentences from Latin script (Arabizi) to Arabic script.
    
    STRICT RULES:
    1. You MUST return exactly {len(lines_list)} lines.
    2. Format: Numbered list (1. ..., 2. ...).
    3. Do not translate meaning, just transliterate to Arabic script.
    4. Do not stop until all {len(lines_list)} are done.
    
    Input:
    {formatted_input}
    
    Output:
    """
    
    # Infinite loop that only breaks on success
    while True:
        try:
            response = model.generate_content(prompt)
            text_response = response.text.strip()
            
            results = []
            for line in text_response.split('\n'):
                clean_line = re.sub(r'^\d+\.?\s*', '', line).strip()
                if clean_line:
                    results.append(clean_line)
            
            # Validation success
            if len(results) >= len(lines_list) - 5:
                # Fix length mismatches
                if len(results) > len(lines_list):
                    return results[:len(lines_list)]
                elif len(results) < len(lines_list):
                    results.extend([""] * (len(lines_list) - len(results)))
                return results
            else:
                print(f"   [Batch {batch_index}] Warning: Count mismatch. Retrying...")
                time.sleep(5)
                
        except Exception as e:
            print(f"   [Batch {batch_index}] Error encountered: {e}")
            
            # CHECK: Is it the internet?
            if not is_connected():
                wait_for_internet()
                continue # Retry immediately after reconnection
            
            # If internet is fine, it might be a Quota or API error. Wait longer.
            print("   [Batch {batch_index}] Retrying in 60 seconds...")
            time.sleep(60)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Initialization...")
    
    # Load Input Data
    try:
        df = pd.read_csv(INPUT_FILE, header=None, on_bad_lines='skip')
        df['clean_latin'] = df[1].apply(clean_text)
        df = df[df['clean_latin'] != ""]
        df = df.reset_index(drop=True)
        all_latin = df['clean_latin'].tolist()
        total_rows = len(all_latin)
        
        # --- SMART RESUME LOGIC ---
        start_index = 0
        
        # Check if output file exists to resume progress
        if os.path.exists(OUTPUT_FILE):
            try:
                existing_df = pd.read_csv(OUTPUT_FILE)
                rows_done = len(existing_df)
                if rows_done > 0:
                    start_index = rows_done
                    print(f"   🔄 Found existing file with {rows_done} rows. Resuming from row {rows_done+1}...")
            except pd.errors.EmptyDataError:
                # File exists but is empty
                pd.DataFrame(columns=['input_text', 'target_text']).to_csv(OUTPUT_FILE, index=False)
        else:
            # Create new file with header
            pd.DataFrame(columns=['input_text', 'target_text']).to_csv(OUTPUT_FILE, index=False)
            print(f"   🆕 Created new output file: {OUTPUT_FILE}")

        if start_index >= total_rows:
            print("   ✅ Dataset is already fully processed!")
            exit()

        # --- PROCESSING LOOP ---
        total_batches = (total_rows // BATCH_SIZE) + 1
        
        # We start loop from the 'start_index' we calculated above
        for i in range(start_index, total_rows, BATCH_SIZE):
            batch_latin = all_latin[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            print(f"   Processing Batch {batch_num}/{total_batches} ({len(batch_latin)} items)...")
            
            # Get Translations (Robust Function)
            translations = process_mega_batch_robust(batch_latin, batch_num)
            
            # Save Batch
            batch_df = pd.DataFrame({
                'input_text': batch_latin,
                'target_text': translations
            })
            
            # Filter bad rows
            batch_df = batch_df[batch_df['target_text'] != "ERROR"]
            batch_df = batch_df[batch_df['target_text'] != ""]

            # Append to CSV
            batch_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            print(f"   -> Saved. Progress: {min(i + BATCH_SIZE, total_rows)}/{total_rows}")
            
            # Wait
            if i + BATCH_SIZE < total_rows:
                print(f"   Waiting {DELAY_SECONDS}s...")
                time.sleep(DELAY_SECONDS)

        print("\n🎉 DONE! Full dataset processed and saved.")

    except Exception as e:
        print(f"Critical Setup Error: {e}")