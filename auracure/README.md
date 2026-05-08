# ❤️ AuraEcho+ — Cardiac AI Decision Support System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-powered cardiac risk assessment that works online AND offline**

*Built for clinical environments where internet is unreliable and patient privacy is non-negotiable*

[🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#️-architecture) • [✨ Features](#-key-features) • [🔐 Security](#-security-notes)

</div>

---

## 🩺 What Is AuraEcho+?

AuraEcho+ is a **clinical decision support system** for cardiac risk assessment. A doctor or nurse enters patient vitals (age, blood pressure, cholesterol, ECG data, etc.), and the system:

1. **Scores cardiac risk** — Low / Medium / High using a trained Random Forest model
2. **Finds similar past patients** — KNN similarity engine matches against historical cases
3. **Generates AI diagnosis** — Llama3 (offline) or Groq/GPT-4o (online) analyses the case
4. **Provides clinical guidelines** — ACC/AHA treatment recommendations
5. **Syncs automatically** — saves locally when offline, pushes to Firebase when reconnected
6. **Enforces role-based access** — Doctor, Nurse, Admin, and Viewer roles with granular permissions

---

## ✨ Key Features

| Feature | Technology | Works Offline? |
|---------|-----------|---------------|
| 🧠 Risk Scoring | Random Forest (sklearn) | ✅ Always |
| 🔍 Similar Cases | KNN + Cosine Similarity | ✅ Always |
| 🤖 AI Diagnosis | Ollama + Llama3 | ✅ Yes |
| ☁️ Cloud AI | Groq API (300+ tok/s) | ❌ Online only |
| 🗄️ Local Storage | SQLite | ✅ Always |
| ☁️ Cloud Sync | Firebase Firestore | ❌ Online only |
| 🔐 Auth | Role-based (Doctor/Nurse/Admin) | ✅ Always |
| 📋 Guidelines | ACC/AHA embedded | ✅ Always |

---

## 🚀 Quick Start

### Option 1 — One-command setup (recommended)
```bash
git clone https://github.com/yourusername/auraecho-plus.git
cd auraecho-plus
bash setup.sh          # installs everything, trains model, creates sample data
make run               # launches the app