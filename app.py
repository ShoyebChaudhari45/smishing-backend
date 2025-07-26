from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("signatures.txt", "r") as f:
    signatures = set(line.strip().lower() for line in f if line.strip())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "").lower()
        matched_keywords = [kw for kw in signatures if kw in message]
        is_spam = bool(matched_keywords)

        return jsonify({
            "spam": is_spam,
            "reason": f"Matched keywords: {', '.join(matched_keywords)}" if is_spam else "No suspicious keywords found"
        }), 200

    except Exception as e:
        return jsonify({
            "spam": False,
            "reason": f"Error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
