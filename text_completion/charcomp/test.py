import torch
from main import CharLSTM, test_completion
from main import VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DEVICE, CHAR_TO_IDX, IDX_TO_CHAR


def load_model(path="char_lstm_best_model.pth"):
    model = CharLSTM(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main():
    model = load_model()
    test_contexts = [
        "To build a lightw",  # lightweight
        "The quick brown fox ju",  # fox
        "Hi All, Sorry for the delay in addressing this is",  # issue
        "The meeting is sch",  # scheduled
        "Hi all, Due to Hurricane Milton, I am canc",  # canceling
        "The results are in and the wi",  # Should not be a-z.
    ]

    for test_context in test_contexts:
        completion = test_completion(model, test_context, max_len=50, n_break=3)
        print(f"Context: {test_context}")
        print(f"Completion: {completion}\n")


if __name__ == "__main__":
    main()