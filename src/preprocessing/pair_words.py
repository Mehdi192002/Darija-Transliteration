import pandas as pd
import google.generativeai as genai
import time
import os
import json
import re

# --- CONFIGURATION ---
API_KEY = 'AIzaSyDVxg3meldtTVBtHkj_ldRzdJOY3JX0ntk'  # <--- PASTE YOUR KEY HERE
INPUT_FILE = os.path.join('..', '..', 'data', 'processed', 'darija_final_dataset.csv') # Must have 2 columns: Latin, Arabic
OUTPUT_FILE = os.path.join('..', '..', 'data', 'interim', 'darija_word_pairs.csv')
BATCH_SIZE = 8  # Reduced batch size slightly for 12B model consistency

genai.configure(api_key=API_KEY)

# Using the Instruction Tuned ('it') version for better rule following
model = genai.GenerativeModel('gemma-3-12b-it') 

def clean_text(text):
    if not isinstance(text, str): return ""
    return " ".join(text.split())

def get_aligned_pairs(batch_data):
    """
    batch_data is a list of tuples: [("Salam cv", "سلام صافي"), ...]
    """
    
    # Construct a clear, structured input for the model
    formatted_input = ""
    for idx, (lat, arb) in enumerate(batch_data):
        formatted_input += f"--- PAIR {idx+1} ---\nLATIN: {lat}\nARABIC: {arb}\n"

    prompt = f"""
    Role: Word Aligner.
    Task: Align the Latin words to their corresponding Arabic words from the provided text.
    
    INSTRUCTIONS:
    1. Read the "LATIN" sentence and the "ARABIC" sentence.
    2. Match every Latin word to its exact Arabic counterpart found in the "ARABIC" line.
    3. IMPORTANT: Do not translate. You must COPY the Arabic word exactly as it appears in the input.
    4. If the Latin word has no match (or is an emoji), ignore it.
    
    INPUT DATA:
    {formatted_input}
    
    OUTPUT FORMAT:
    Return ONLY a raw JSON list of objects. Do not use Markdown code blocks.
    [
      {{"latin": "latin_word", "arabic": "arabic_word_from_text"}},
      ...
    ]
    """
    
    try:
        # We use standard generation (no forced MIME type) to avoid API compatibility issues
        response = model.generate_content(prompt)
        text = response.text
        
        # --- CLEANUP ---
        # Gemma often adds ```json at the start. We strip it.
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Parse
        data = json.loads(text)
        return data
        
    except json.JSONDecodeError:
        print(f"   ⚠️ Alignment Warning: Model output invalid JSON. Skipping batch.")
        return []
    except Exception as e:
        print(f"   ⚠️ Alignment Error: {e}")
        return []

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    print("1. Loading Dataset...")
    try:
        # Load Data
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Error: {INPUT_FILE} not found.")
            exit()

        df = pd.read_csv(INPUT_FILE, header=None, on_bad_lines='skip')
        
        # Ensure we have at least 2 columns and drop empty rows
        if df.shape[1] < 2:
            print("❌ Error: Input CSV must have at least 2 columns (Latin, Arabic).")
            exit()
            
        df = df.dropna(subset=[0, 1])
        
        # Zip columns into pairs: (Latin, Arabic)
        sentence_pairs = list(zip(df[0], df[1]))
        print(f"   Loaded {len(sentence_pairs)} pairs. Using gemma-3-12b-it.")
        
        # Setup Output
        if not os.path.exists(OUTPUT_FILE):
             pd.DataFrame(columns=['latin', 'arabic']).to_csv(OUTPUT_FILE, index=False)

        # Loop
        for i in range(0, len(sentence_pairs), BATCH_SIZE):
            batch = sentence_pairs[i : i + BATCH_SIZE]
            print(f"   Aligning Batch {i}...", end=" ")
            
            try:
                aligned_words = get_aligned_pairs(batch)
                
                if aligned_words and isinstance(aligned_words, list):
                    batch_df = pd.DataFrame(aligned_words)
                    
                    if 'latin' in batch_df.columns and 'arabic' in batch_df.columns:
                        # Basic cleaning
                        batch_df['latin'] = batch_df['latin'].apply(clean_text)
                        batch_df['arabic'] = batch_df['arabic'].apply(clean_text)
                        
                        # Remove duplicates and save
                        batch_df = batch_df.drop_duplicates()
                        batch_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                        print(f"✅ Saved {len(batch_df)} pairs.")
                    else:
                        print("⚠️ JSON keys missing.")
                else:
                    print("❌ No valid data.")
            
            except Exception as e:
                print(f"❌ Batch Error: {e}")

            # Gemma 3 12B is larger than Flash, so we wait 4 seconds to avoid 429 errors
            time.sleep(4)

        print("\n🎉 Done! Word alignment complete.")
        
    except Exception as e:
        print(f"Critical System Error: {e}")