from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from load_model_gpt_lstm import load_gpt_model, load_char_lstm, load_original_gpt2
from load_model_gpt_lstm import char_completion, predict_text
import os

# Initialize Flask app
app = Flask(__name__, static_folder="resources")
CORS(app)

# Load model and tokenizer
base_dir = "/mnt/c/Users/anguangyan/Study/text_completion/models"
sub_folder = "fine-tuning"
lstm_sub_folder = "lstm_char"
# TODO: Choose the model here
# gpt_model, tokenizer, device = load_gpt_model(base_dir, sub_folder)
gpt_model, tokenizer, device = load_original_gpt2()
lstm_model = load_char_lstm(base_dir, lstm_sub_folder, device)
# End TODO

# Define model's maximum token limit
MAX_TOKEN_LIMIT = gpt_model.config.n_positions  # Typically 1024 for GPT-2
print(f"Model's maximum token limit: {MAX_TOKEN_LIMIT}")


@app.route("/")
def index():
    return send_from_directory("resources", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("resources", path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_text = data.get("text", "")
    max_new_tokens = data.get("max_new_tokens", 10)

    top_k = data.get("top_k", 2)

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    try:
        # Try the lstm model first
        input_striped = input_text.strip()
        prob_c, completion = char_completion(lstm_model, input_striped, device, max_length_char=max_new_tokens, n_break=1)
        if completion != "":
            return jsonify({"predictions": [completion],
                            "prob": [prob_c]})
        # Tokenize the input text and check length
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_token_count = len(input_ids[0])

        if input_token_count + max_new_tokens > MAX_TOKEN_LIMIT:
            max_input_length = MAX_TOKEN_LIMIT - max_new_tokens
            return jsonify({
                "error": f"Input text is too long. Maximum input length is {max_input_length} tokens.",
                "max_input_length": max_input_length
            }), 400

        # Generate predictions
        predictions = predict_text(gpt_model, tokenizer, input_text, device, max_length=max_new_tokens, top_k=top_k)
        prob_g = [p[0] for p in predictions]
        predictions = [p[1] for p in predictions]
        return jsonify({"predictions": predictions,
                        "prob": prob_g})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
