from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("..", "..", "models", "v2")

print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Model loaded on {device}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def transliterate_word_only(arabic_word):
    """
    Sends a SINGLE word to the AI model.
    """
    inputs = tokenizer(arabic_word, return_tensors="pt").to(device)
    
    # Generate (Beam search for quality)
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=20, # Words are short
        num_beams=4, 
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def process_paragraph(text):
    """
    Splits paragraph into words, handles rules (numbers, single chars),
    and reconstructs the sentence.
    """
    # 1. Split by space
    raw_tokens = text.split()
    final_output = []
    
    # Buffer to hold single characters (like 'w' or 'f')
    prefix_buffer = ""

    for token in raw_tokens:
        # CLEANUP: Remove common punctuation attached to words for better matching
        # (Optional: keep punctuation if you want, but models prefer clean words)
        clean_token = token.strip()
        
        # RULE 1: If it is a Number (digits), keep it AS IS.
        # We check if the token contains digits (e.g., "2024" or "100dh")
        if re.search(r'\d', clean_token):
            # If we had a prefix waiting (e.g. "w 2000"), attach it? 
            # Usually better to output prefix + space + number, or just clear prefix.
            # Let's attach it to be safe:
            if prefix_buffer:
                # Transliterate prefix separately or just output raw? 
                # Usually single char arabic -> latin single char
                # Let's just append the raw arabic prefix to the number for now
                final_output.append(prefix_buffer + clean_token)
                prefix_buffer = ""
            else:
                final_output.append(clean_token)
            continue

        # RULE 2: Single Character (و , f, b, etc.)
        # If the word is just 1 letter, hold it and add to next word.
        if len(clean_token) == 1:
            prefix_buffer += clean_token
            continue

        # RULE 3: Standard Word processing
        # Combine with any waiting prefix
        word_to_translate = prefix_buffer + clean_token
        
        # Reset buffer
        prefix_buffer = ""

        # Send to AI Model
        translated_word = transliterate_word_only(word_to_translate)
        final_output.append(translated_word)

    # Edge case: If sentence ends with a single character
    if prefix_buffer:
        final_output.append(transliterate_word_only(prefix_buffer))

    # Join back into a sentence
    return " ".join(final_output)

# --- MAIN LOOP ---
print("\n--- Darija Paragraph Translator ---")
print("Type a full sentence/paragraph in Arabic script.")
print("Type 'q' to quit.\n")

while True:
    user_input = input("Arabic Input: ")
    if user_input.lower() in ['q', 'quit', 'exit']:
        break
    
    if not user_input.strip():
        continue

    result = process_paragraph(user_input)
    print(f"Latin Output: {result}")
    print("-" * 30)