# ❤️ AuraEcho+ — Cardiac AI Decision Support System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-powered cardiac risk assessment that works online AND offline**

*Built for clinical environments where internet is unreliable and patient privacy is non-negotiable*

[🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#️-architecture) • [🖥️ Screenshots](#️-screenshots) • [🤝 Contributing](#-contributing)

</div>

---

## 🩺 What Is AuraEcho+?

AuraEcho+ is a **clinical decision support system** for cardiac risk assessment. A doctor or nurse enters patient vitals (age, blood pressure, cholesterol, ECG data, etc.), and the system:

1. **Scores cardiac risk** — Low / Medium / High using a trained Random Forest model
2. **Finds similar past patients** — KNN similarity engine matches against 500 historical cases  
3. **Generates AI diagnosis** — Llama3 (offline) or Groq/GPT-4o (online) analyses the case
4. **Provides clinical guidelines** — ACC/AHA 2023 treatment recommendations auto-loaded
5. **Checks drug interactions** — flags dangerous cardiac medication combinations
6. **Syncs automatically** — saves locally when offline, pushes to Firebase when reconnected

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
| 🔐 Auth | Role-based (Doctor/Nurse) | ✅ Always |
| 📋 Guidelines | ACC/AHA 2023 embedded | ✅ Always |

---

## 🚀 Quick Start

### Option 1 — One-command setup (recommended)
```bash
git clone https://github.com/yourusername/auraecho-plus.git
cd auraecho-plus
bash setup.sh          # installs everything
make run               # launches the app
```

### Option 2 — Manual setup
```bash
# 1. Clone the repo
git clone https://github.com/yourusername/auraecho-plus.git
cd auraecho-plus

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
nano .env   # add your GROQ_API_KEY

# 5. Train the model
python3 -c "from core.risk_model import train_model; train_model()"

# 6. Launch
streamlit run app.py
```

### Option 3 — Offline only (no API keys needed)
```bash
bash setup.sh
ollama pull llama3   # 4GB download — enables offline AI
make run
```

Open **http://localhost:8501** in your browser.

---

## 🔑 Demo Credentials

| Role | Username | Password | Access Level |
|------|----------|----------|-------------|
| 👨‍⚕️ Doctor | `admin_doctor` | `Doctor@123` | Full — AI diagnosis, export, user management |
| 👩‍⚕️ Nurse | `nurse_demo` | `Nurse@123` | Restricted — vitals entry, risk scores only |

---

## 🏗️ Architecture

```
auraecho-plus/
│
├── app.py                    # 🚀 Master entry point
│
├── ui/                       # 🎯 Streamlit UI components
│   ├── sidebar.py            #    Patient data entry form
│   ├── results_panel.py      #    Diagnosis cards + similar cases
│   └── dashboard.py          #    Analytics charts (Plotly)
│
├── ai/                       # 🤖 AI backends (same prompt, two models)
│   ├── prompt_builder.py     #    Structures patient data → LLM prompt
│   ├── offline_ai.py         #    Ollama + Llama3 (local, private)
│   └── online_ai.py          #    Groq → OpenAI (cloud, faster)
│
├── core/                     # 🧠 ML engine
│   ├── preprocess.py         #    Clean + scale patient features
│   ├── risk_model.py         #    Random Forest risk classifier
│   ├── similarity.py         #    KNN similar cases engine
│   └── mode_detector.py      #    Internet connectivity check
│
├── database/                 # 🗄️ Data persistence
│   ├── local_db.py           #    SQLite (offline-first)
│   └── cloud_db.py           #    Firebase Firestore (sync)
│
├── services/                 # ⚙️ Business logic services
│   ├── auth_service.py       #    Login + role-based access
│   ├── api_service.py        #    ACC/AHA guidelines + drug checker
│   └── sync_service.py       #    Offline → cloud sync (background)
│
├── utils/                    # 🔧 Shared utilities
│   ├── constants.py          #    All configuration values
│   ├── helpers.py            #    Shared functions
│   └── validators.py         #    Input validation
│
└── data/
    ├── heart_data.csv        # 500-row Cleveland Heart Dataset
    └── sample_input.json     # Demo patient payload
```

### Data Flow Diagram

```
Patient Form Input
      │
      ▼
[validators.py] ──► Validate inputs
      │
      ▼
[preprocess.py] ──► Clean + Scale (1×13 array)
      │
      ├──────────────────┐
      ▼                  ▼
[risk_model.py]    [similarity.py]
Random Forest      KNN Engine
Risk Level +       Top 3 Similar
Confidence %       Historical Cases
      │                  │
      └────────┬─────────┘
               ▼
    [mode_detector.py]
    Online?  /  \  Offline?
            /    \
   [online_ai]  [offline_ai]
   Groq/GPT-4   Ollama/Llama3
           \    /
            \  /
             ▼
     [prompt_builder.py]
     Structured LLM Prompt
             │
             ▼
       AI Response
             │
      ┌──────┴───────┐
      ▼              ▼
[local_db.py]   [cloud_db.py]
  SQLite         Firebase
  (always)       (if online)
```

---

## 📋 Clinical Features

### Risk Scoring
The Random Forest model uses 13 Cleveland Heart Disease features:

| Feature | Clinical Meaning |
|---------|----------------|
| `age` | Patient age in years |
| `sex` | Biological sex (0=Female, 1=Male) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL |
| `restecg` | Resting ECG result (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression (exercise vs rest) |
| `slope` | Slope of peak ST segment |
| `ca` | Number of major vessels (0-3) |
| `thal` | Thalassemia type (0-3) |

### AI Diagnosis Sections
Every AI response is structured into:
- 🔍 **Clinical Assessment** — overall cardiac picture
- ⚠️ **Key Risk Indicators** — top concerning findings
- 🔮 **Future Symptoms** — what to watch for
- 💊 **Treatment Recommendations** — immediate/short/long term
- 🏥 **Referral & Follow-up** — who, when, and what tests
- 📋 **Patient Education** — plain-language talking points

---

## 🔧 Configuration

### Environment Variables (`.env`)

```env
# Online AI — Groq (primary, ultra-fast)
GROQ_API_KEY=gsk_your_key_here

# Online AI — OpenAI (fallback)
OPENAI_API_KEY=sk-your_key_here

# Firebase Cloud Sync
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_CREDENTIALS_PATH=data/firebase_credentials.json

# App settings
LOG_LEVEL=INFO
```

### Getting API Keys
- **Groq** (free): [console.groq.com](https://console.groq.com) → Create API Key
- **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)
- **Firebase**: [console.firebase.google.com](https://console.firebase.google.com) → Project Settings → Service Accounts

---

## 🧪 Running Tests

```bash
make test              # All tests with coverage report
make test-fast         # Faster (no coverage)
make test-single FILE=tests/test_risk_model.py
```

Test coverage targets:
- `tests/test_similarity.py`  — KNN engine
- `tests/test_risk_model.py`  — Risk scorer
- `tests/test_ai.py`          — AI responses
- `tests/test_auth.py`        — Login + permissions
- `tests/test_validators.py`  — Input validation

---

## 📊 Model Performance

Trained on [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (303 original + synthetic expansion to 500):

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| F1 Score | ~84% |
| ROC-AUC | ~91% |
| Precision | ~83% |
| Recall | ~86% |

---

## 🛠️ Make Commands Reference

```bash
make help          # Show all commands
make run           # Start the app
make setup         # First-time setup
make train         # Retrain ML model
make test          # Run tests + coverage
make clean         # Remove cache files
make status        # System health check
make demo-data     # Load sample patient for demo
make ollama-pull   # Download llama3 model
make db-stats      # Database statistics
make db-export     # Export patients to CSV
```

---

## 🔐 Security Notes

- ✅ Passwords hashed with **bcrypt** (never stored plain-text)
- ✅ Session tokens expire after **8 hours**
- ✅ Role-based access control enforced server-side
- ✅ Patient data stays on-device in offline mode
- ✅ `.env` file excluded from Git via `.gitignore`
- ⚠️ Change default passwords before production use
- ⚠️ This is a **clinical decision support tool** — not a replacement for physician judgment

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) — UCI ML Repository
- [ACC/AHA 2023 Guidelines](https://www.acc.org/guidelines) — Clinical content source
- [Ollama](https://ollama.ai) — Local LLM inference
- [Groq](https://groq.com) — Ultra-fast cloud inference
- [Streamlit](https://streamlit.io) — UI framework

---

<div align="center">
Built with ❤️ for better cardiac care

<sub>⚠️ AuraEcho+ is for clinical decision support only. Always consult a qualified physician.</sub>
</div>