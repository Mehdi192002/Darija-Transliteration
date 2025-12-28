import pandas as pd
import random

# --- CONFIGURATION ---
OUTPUT_FILE = os.path.join('..', '..', 'data', 'interim', 'synthetic_dataset.csv')
NUM_SAMPLES = 5000  # How many fake words to generate

# 1. Define the Strict Mappings
# This includes your specific rules + standard Darija mappings
CHAR_MAP = {
    # --- YOUR SPECIAL RULES ---
    '4': 'غ',
    '3': 'ع',
    '5': 'خ',
    '9': 'ق',
    '2': 'أ',
    'x': 'ش',
    
    # --- STANDARD MAPPINGS ---
    'a': 'ا',
    'b': 'ب',
    't': 'ت',
    'th': 'ث', # multichar handling is tricky in simple loop, simplified here
    'j': 'ج',
    'H': 'ح', # Standard 7 is often H, but we stick to single chars first
    '7': 'ح',
    'd': 'د',
    'r': 'ر',
    'z': 'ز',
    's': 'س',
    'f': 'ف',
    'k': 'ك',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'h': 'ه',
    'w': 'و',
    'y': 'ي',
    'o': 'و', # 'o' often maps to waw
    'e': 'ي', # 'e' often maps to ya
    'u': 'و',
    'i': 'ي',
}

# List of keys to pick from
KEYS = list(CHAR_MAP.keys())

def generate_fake_pair():
    # 1. Choose a random length (3 to 8 characters)
    length = random.randint(3, 8)
    
    latin_word = ""
    arabic_word = ""
    
    for _ in range(length):
        # Pick a random character key
        # We give slightly higher weight to your special numbers to ensure they appear often
        if random.random() < 0.3: # 40% chance to pick a special number/char
            char = random.choice(['4', '3', '5', '9', '2', 'x'])
        else:
            char = random.choice(KEYS)
            
        latin_word += char
        arabic_word += CHAR_MAP[char]
        
    return latin_word, arabic_word

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Generating {NUM_SAMPLES} synthetic words...")
    
    data = []
    for _ in range(NUM_SAMPLES):
        lat, arb = generate_fake_pair()
        data.append({'latin': lat, 'arabic': arb})
        
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV (header=False to match your existing training format)
    df.to_csv(OUTPUT_FILE, index=False, header=False)
    
    print(f"✅ Saved to {OUTPUT_FILE}")
    print("\nSample Data Generated:")
    print(df.head(10))