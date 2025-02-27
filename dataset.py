import json
import torch
from torch.utils.data import Dataset, DataLoader

# Define maximum sequence length
MAX_SEQ_LEN = 128  

# ✅ Load training data
with open("train_data.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

# ✅ Load validation data
with open("validation_data.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"✅ Loaded {len(train_data)} training samples!")
print(f"✅ Loaded {len(val_data)} validation samples!")

# Define a Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, dataset, max_length=MAX_SEQ_LEN):
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        input_ids = data["input_ids"]
        target_ids = data["target_ids"]

        # Pad sequences to max_length
        input_ids = input_ids[:self.max_length] + [0] * (self.max_length - len(input_ids))
        target_ids = target_ids[:self.max_length] + [0] * (self.max_length - len(target_ids))

        # Create attention masks
        input_mask = [1 if token != 0 else 0 for token in input_ids]
        target_mask = [1 if token != 0 else 0 for token in target_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
        }

# Create Dataset objects
train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)  # Validation dataset

# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # For validation

print(f"✅ DataLoader created! Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
