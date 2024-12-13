import os
import re
import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import string


VOCAB = list(string.ascii_lowercase + string.ascii_uppercase + " .,!?")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
MAX_SEQ_LEN = 100
EMBED_DIM = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 3


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


def find_last_number(s):
    matches = re.findall(r'_\d+_', s)
    if matches:
        # Extract the number from the last match
        return int(matches[-1][1:-1])
    return None


def find_last_checkpoint(base_dir, sub_folder):
    model_path = os.path.join(base_dir, sub_folder)
    if not os.path.isdir(model_path):
        raise ValueError(f"Invalid subfolder '{sub_folder}'. Directory does not exist: {model_path}")

    checkpoints = [f for f in os.listdir(model_path) if f.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in subfolder '{sub_folder}'")

    last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(model_path, last_checkpoint)


def load_gpt_model(base_dir, sub_folder, device=None):
    """
    Load a model from a specified subfolder.
    """
    model_path = find_last_checkpoint(base_dir, sub_folder)

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Move model to appropriate device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def load_original_gpt2(device=None):
    """
    Load the original GPT-2 model.
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Move model to appropriate device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def load_char_lstm(base_dir, sub_folder, device=None):
    """
    Load a character-level LSTM model from a specified subfolder.
    """
    model_path = os.path.join(base_dir, sub_folder)
    checkpoints = [f for f in os.listdir(model_path) if f.endswith(".pth")]
    numbers = [find_last_number(f) for f in checkpoints]
    last_checkpoint = checkpoints[numbers.index(max(numbers))]
    model_path = os.path.join(model_path, last_checkpoint)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CharLSTM(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def predict_text(model, tokenizer, input_text, device, max_length=50, top_k=2):
    """
    Generate predictions from the model given input text.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=len(inputs.input_ids[0]) + max_length,
        num_return_sequences=top_k,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
    )
    predictions = [(1,tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)) for output in outputs]
    return predictions


def char_completion(model, context, device, max_length_char=30, n_break=1):
    if max_length_char > MAX_SEQ_LEN - 20:
        max_length_char = MAX_SEQ_LEN - 20
    if len(context) + max_length_char > MAX_SEQ_LEN:
        context = context[-MAX_SEQ_LEN + max_length_char:]
    model.eval()
    input_seq = [CHAR_TO_IDX.get(c, 0) for c in context]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    generated_text = ""
    n_punc = 0
    prob1 = None
    with torch.no_grad():
        for _ in range(max_length_char):
            logits = model(input_tensor)
            last_char_logits = logits[0, -1, :]
            next_char_idx = torch.argmax(last_char_logits).item()
            next_char = IDX_TO_CHAR[next_char_idx]
            if prob1 is None:
                prob1 = torch.softmax(last_char_logits, dim=0)[next_char_idx].item()
            if next_char in [" ", ".", "!", "?"]:
                n_punc += 1
            if n_punc >= n_break:
                break
            generated_text += next_char
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_char_idx]], device=device)], dim=1)
    return prob1, generated_text


def main():
    # Define base directory and subfolder for model
    base_directory = "/mnt/c/Users/anguangyan/Study/text_completion/models"
    subfolder_name = ["distillation", "fine-tuning", "pruning", "quantization"][2]

    # Input text
    input_text = "The quick brown fox jumps over"

    # Load the model
    model, tokenizer, device = load_gpt_model(base_directory, subfolder_name)

    # Perform prediction
    print("Generating prediction...")
    prediction = predict_text(model, tokenizer, input_text, device)
    print(f"Input: {input_text}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
