import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import string
from tqdm import tqdm

# Define constants
VOCAB = list(string.ascii_lowercase + string.ascii_uppercase + " .,!?")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
MAX_SEQ_LEN = 100
OVERLAP = MAX_SEQ_LEN // 3
BATCH_SIZE = 1024
EMBED_DIM = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 3
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOP_PATIENCE = 5

# Dataset class
class CharDataset(Dataset):
    def __init__(self, data, char_to_idx, max_seq_len=MAX_SEQ_LEN, overlap=OVERLAP):
        self.data = []
        self.char_to_idx = char_to_idx
        self.max_seq_len = max_seq_len
        self.overlap = overlap

        # Split text into chunks with overlap
        for text in data:
            text = text.replace("\n", " ")
            for i in range(0, len(text) - self.overlap, self.max_seq_len - self.overlap):
                chunk = text[i:i + self.max_seq_len]
                if len(chunk) == self.max_seq_len:
                    self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = [self.char_to_idx.get(c, 0) for c in text[:-1]]
        targets = [self.char_to_idx.get(c, 0) for c in text[1:]]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# Define the updated LSTM model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits


# Testing function
def test_completion(model, context, max_len=30, n_break=1):
    model.eval()
    input_seq = [CHAR_TO_IDX.get(c, 0) for c in context]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated_text = context
    n_punc = 0
    p1 = False
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_tensor)
            last_char_logits = logits[0, -1, :]
            next_char_idx = torch.argmax(last_char_logits).item()
            next_char = IDX_TO_CHAR[next_char_idx]
            if not p1:
                print(next_char, torch.softmax(last_char_logits, dim=0)[next_char_idx].item())
                p1 = True
            if next_char in [" ", ".", "!", "?"]:
                n_punc += 1
            if n_punc >= n_break:
                break
            generated_text += next_char
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_char_idx]], device=DEVICE)], dim=1)
    return generated_text


if __name__ == '__main__':
    # Load and preprocess dataset
    print("Loading dataset...")
    dataset = load_dataset("stas/openwebtext-10k", split="train")
    texts = [sample["text"].replace("\n", " ") for sample in tqdm(dataset, desc="Processing dataset") if
             "text" in sample]
    char_dataset = CharDataset(texts, CHAR_TO_IDX)
    train_loader = DataLoader(char_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Initialize the model
    model = CharLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Training the model...")
    best_loss = float('inf')
    patience = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, VOCAB_SIZE)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss /= len(train_loader)
        # print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
            torch.save(model.state_dict(), "char_lstm_best_model.pth")  # Save the best model
            # print("Model improved and saved!")
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{EARLY_STOP_PATIENCE}")
            if patience >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    # Test the model
    test_contexts = [
        "To build a lightw",  # lightweight
        "The quick brown f",  # fox
        "Hi All, Sorry for the delay in addressing this is",  # issue
        "The meeting is sch",  # scheduled
        "Hi all, Due to Hurricane Milton, I am can",  # canceling
        "The results are in and the winner is",  # Should not be a-z.
    ]

    for test_context in test_contexts:
        completion = test_completion(model, test_context)
        print(f"Context: {test_context}")
        print(f"Completion: {completion}\n")
