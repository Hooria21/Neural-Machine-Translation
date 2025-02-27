from datasets import load_dataset, DatasetDict
import sentencepiece as spm
import json

# Load Arabic-English dataset
dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "ara-eng", trust_remote_code=True)

# Combine all text data for tokenizer training
all_text = []

for split in ["validation", "test"]:  # Using validation as training data
    for example in dataset[split]:
        # Ensure text is properly formatted
        src = example["sourceString"].strip()
        tgt = example["targetString"].strip()
        
        if src and tgt:  # Avoid empty lines
            all_text.append(src)
            all_text.append(tgt)

# Save to a text file for tokenizer training
with open("all_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))  # Faster than writing line-by-line

# Check available splits
print(dataset)  # This will show all available dataset splits

# Since there is no "train" split, split the validation set into train and validation
train_test_split = dataset["validation"].train_test_split(test_size=0.2, seed=42, shuffle=True)

# Create a new dataset dictionary with train, validation, and test sets
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"],  # This is now a smaller validation set
    "test": dataset["test"]
})

# Access the newly created splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

print(f"Train size: {len(train_data)}, Validation size: {len(validation_data)}, Test size: {len(test_data)}")

# Train the tokenizer using SentencePiece
spm.SentencePieceTrainer.Train(
    input="all_text.txt",
    model_prefix="tokenizer",
    vocab_size=30_000,
    character_coverage=1.0,  # Full character coverage
    model_type="bpe",  # Can also try 'unigram', 'word', or 'char'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3      
)

print("SentencePiece Tokenizer training complete! ✅")  

# Load trained tokenizer
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")

# Test it on a sample sentence
test_sentence = "السلام عليكم! How are you?"
encoded = sp.encode(test_sentence, out_type=str)

print("Tokenized output:", encoded)
print("Token IDs:", sp.encode(test_sentence))

def tokenize_function(example):
    return {
        "input_ids": sp.encode(example["sourceString"]),  # Tokenize source text
        "target_ids": sp.encode(example["targetString"])  # Tokenize target text
    }

# Apply tokenization to the full dataset
tokenized_dataset = dataset.map(tokenize_function, remove_columns=["sourceString", "targetString"])

# Show an example
print(tokenized_dataset["train"][0])

# Convert dataset to list format for JSON storage
train_data = list(tokenized_dataset["train"])
validation_data = list(tokenized_dataset["validation"])
test_data = list(tokenized_dataset["test"])

# Save datasets as JSON files
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("validation_data.json", "w", encoding="utf-8") as f:
    json.dump(validation_data, f, ensure_ascii=False, indent=4)

with open("test_data.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("✅ Tokenized data saved: train_data.json, validation_data.json, test_data.json")

