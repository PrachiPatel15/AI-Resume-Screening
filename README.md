# AI Resume Screening

![AI Resume Screening](https://img.shields.io/badge/AI-Powered-blue.svg)

## 🚀 Overview
AI Resume Screening is a **Streamlit-based web application** that leverages **AI and NLP** to analyze resumes and extract valuable insights such as **skills, experience, and qualifications**. This project helps recruiters streamline the hiring process by providing **automated resume analysis and AI-powered Q&A**.

## 🌟 Features
- 📄 **Resume Parsing**: Extracts key details from resumes.
- 🤖 **AI-Powered Q&A**: Ask AI anything about a candidate's resume.
- 🎯 **Skill & Experience Matching**: Identifies relevant qualifications for job roles.
- 🏆 **Hiring Decision Assistance**: Provides summarized insights to recruiters.
- 📊 **Streamlit UI**: User-friendly interface for seamless interaction.

## 🏗️ Tech Stack
- **Python** 🐍
- **Streamlit** 🎨 (Frontend)
- **LangChain / Groq** 🤖 (AI-powered Q&A)

## 📂 Project Structure
```
AI-Resume-Screening/
│── app.py                 # Main Streamlit application
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│── models/                # Pretrained models (if applicable)
│── utils/                 # Helper functions
│── qa_system.py           # QnA part
```

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PrachiPatel15/AI-Resume-Screening.git
cd AI-Resume-Screening
```

### 2️⃣ Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## ⚙️ Configuration
- Modify `app.py` to customize AI behavior.
- Use different LLM to check accuracy.
- Do hyperparameter tuning to see the variation.

## 🚀 Deployment
You can deploy this app on **Streamlit Cloud, Hugging Face Spaces, or AWS** using the following methods:
```bash
git push origin main  # Deploy via GitHub Actions (if configured)
```

## 🤝 Contribution
Want to improve this project? Feel free to:
- Open an **issue** 🛠️
- Submit a **pull request** ✨
- Share **feedback** 💡

---
### 🌟 Show Your Support
If you find this project helpful, consider ⭐ starring the repository! 😊

