from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Path to your saved model
MODEL_PATH = "./darija_transliteration_model"

print("Loading your new Darija model...")
try:
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    def transliterate(arabic_text):
        # Prepare input
        inputs = tokenizer(arabic_text, return_tensors="pt").to(device)
        
        # Generate output (Beam search for better quality)
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=40, 
            num_beams=4, 
            early_stopping=True
        )
        
        # Decode result
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n✅ Model Loaded! Type 'q' to exit.")
    print("Try typing: salam, labas, chokran (but in Arabic!)")
    print("-" * 30)

    while True:
        text = input("\nEnter Arabic Darija: ")
        if text.lower() == 'q':
            break
        
        result = transliterate(text)
        print(f"Latin Prediction:   {result}")

except Exception as e:
    print(f"❌ Error: {e}")