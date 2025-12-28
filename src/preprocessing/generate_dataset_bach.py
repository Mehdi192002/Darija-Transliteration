import pandas as pd
import google.generativeai as genai
import time
import re

# --- CONFIGURATION ---
API_KEY = 'AIzaSyDVxg3meldtTVBtHkj_ldRzdJOY3JX0ntk' 
INPUT_FILE = os.path.join('..', '..', 'data', 'raw', 'darija_reels_comments.csv') 
OUTPUT_FILE = 'darija_cleaned_dataset.csv'
BATCH_SIZE = 15  # We process 15 rows per API call
DELAY_SECONDS = 15 # Wait 15s between calls to stay under 5 req/min

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.replace("reel_comment", "")
    text = " ".join(text.split())
    if len(text) < 2: return ""
    return text

def process_batch(lines_list):
    """
    Sends a list of text lines to Gemini in one go.
    """
    # Create a numbered list for the prompt
    formatted_input = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines_list)])
    
    prompt = f"""
    Role: Expert Moroccan Darija Translator.
    Task: Transliterate the list of sentences below from Latin script (Arabizi) to Arabic script.
    
    Strict Output Format:
    Return ONLY a numbered list matching the input. Do not include original text, just the Arabic.
    Example Input:
    1. Slm
    2. Labas
    Example Output:
    1. سلام
    2. لباس

    Input List:
    {formatted_input}
    
    Output List:
    """
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        
        # Parse the numbered list back into a python list
        results = []
        # Split by new lines and remove the "1. ", "2. " prefixes
        for line in text_response.split('\n'):
            # simple regex to remove number and dot at start (e.g., "1. ")
            clean_line = re.sub(r'^\d+\.?\s*', '', line).strip()
            if clean_line:
                results.append(clean_line)
        
        # Safety check: if lengths don't match (rare), pad with None
        if len(results) != len(lines_list):
            print(f"Warning: Batch size mismatch (Sent {len(lines_list)}, got {len(results)})")
            # Fill missing spots with blank strings to keep alignment
            while len(results) < len(lines_list):
                results.append("")
                
        return results
        
    except Exception as e:
        print(f"Batch Error: {e}")
        return [""] * len(lines_list) # Return empty list on failure

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Loading and cleaning data...")
    
    # Load Data
    try:
        df = pd.read_csv(INPUT_FILE, header=None, on_bad_lines='skip')
        df['clean_latin'] = df[1].apply(clean_text)
        df = df[df['clean_latin'] != ""]
        
        # Reset index is crucial for batching logic
        df = df.reset_index(drop=True)
        all_latin = df['clean_latin'].tolist()
        all_arabic = []
        
        total_batches = (len(all_latin) // BATCH_SIZE) + 1
        print(f"   Found {len(df)} rows. Processing in {total_batches} batches...")

        # Loop through data in chunks
        for i in range(0, len(all_latin), BATCH_SIZE):
            batch = all_latin[i : i + BATCH_SIZE]
            batch_idx = (i // BATCH_SIZE) + 1
            
            print(f"   Processing Batch {batch_idx}/{total_batches} ({len(batch)} items)...")
            
            translations = process_batch(batch)
            all_arabic.extend(translations)
            
            # WAIT to respect rate limits (Critical Step)
            if i + BATCH_SIZE < len(all_latin): # Don't sleep after last batch
                print(f"   Waiting {DELAY_SECONDS}s to cool down...")
                time.sleep(DELAY_SECONDS)

        # Assign back to DF
        # Ensure lengths match exactly
        if len(all_arabic) < len(df):
             all_arabic.extend([""] * (len(df) - len(all_arabic)))
        elif len(all_arabic) > len(df):
             all_arabic = all_arabic[:len(df)]

        df['arabic_label'] = all_arabic
        
        # Save
        final_df = df[['clean_latin', 'arabic_label']]
        final_df.columns = ['input_text', 'target_text']
        final_df = final_df[final_df['target_text'] != ""] # Remove any failed rows
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"DONE! Saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Critical Error: {e}")