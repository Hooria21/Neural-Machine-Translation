import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerModel  # Use the correct class name
from dataset import train_loader, val_loader  # Import your DataLoaders

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Model Hyperparameters
VOCAB_SIZE = 30000  # Adjust based on tokenizer vocab size
EMBED_DIM = 256
NUM_HEADS = 8
HIDDEN_DIM = 512
NUM_LAYERS = 6

# ✅ Initialize Transformer Model
model = TransformerModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS).to(device)

# ✅ Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens in loss calculation

# ✅ Enable Automatic Mixed Precision (AMP) for Faster Training
scaler = torch.cuda.amp.GradScaler()

# ✅ Training function
def train_model(model, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  # Enable Mixed Precision
                output = model(input_ids, target_ids)  # Forward pass
                loss = criterion(output.view(-1, VOCAB_SIZE), target_ids.view(-1))

            # ✅ Gradient Scaling for AMP
            scaler.scale(loss).backward()

            # ✅ Clip Gradients (Prevents Exploding Gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ✅ Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

            # ✅ Print progress every 50 batches
            if i % 50 == 0:
                print(f"🟢 Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}")

        # ✅ Validation Step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                
                output = model(input_ids, target_ids)
                loss = criterion(output.view(-1, VOCAB_SIZE), target_ids.view(-1))
                val_loss += loss.item()

            print(f"🔵 Epoch {epoch+1}: Validation Loss = {val_loss / len(val_loader):.4f}")

        # ✅ Save model checkpoint
        torch.save(model.state_dict(), f"transformer_epoch{epoch+1}.pth")
        print(f"💾 Model saved: transformer_epoch{epoch+1}.pth\n")

# ✅ Start Training
train_model(model, train_loader, val_loader, epochs=5)
