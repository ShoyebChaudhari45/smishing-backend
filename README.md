# 📱 Smishing Detection API – Flask Backend

This is the backend for a **Hybrid Smishing (SMS Phishing) Detection System**. It provides a REST API that detects whether an incoming SMS message is smishing (spam) or legitimate.

The detection combines:
- 🛡️ **Signature-based detection** (using `signatures.txt`)
- 🤖 **Machine Learning fallback** (Random Forest Classifier)

---

## ⚙️ How It Works

1. Loads predefined keywords from `signatures.txt`
2. Accepts a POST request at `/predict` with the SMS message
3. Checks the message for any suspicious keywords (signature-based)
4. If no match, uses the trained ML model to predict
5. Returns a JSON response indicating whether it's spam or not

---

## 🧠 Real-World Use Cases

- Integrate into **Android or iOS apps** for real-time smishing detection
- Use in **SMS gateways** to filter messages
- Deploy in **chatbots** to flag spam content
- Build browser extensions or APIs that verify message legitimacy

---

## 🛠️ Tech Stack

- Python 3.x
- Flask
- Flask-CORS
- scikit-learn
- pandas
- joblib

