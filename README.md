# AuthentiScan – AI-Based Digital Document Forgery Detection System

AuthentiScan is an AI-powered web application designed to identify potential forgery in digital documents using deep learning and document analysis techniques. Built with Python and Flask, the system provides a user-friendly interface for uploading and analyzing documents across different categories.

## 🚀 Features

- **AI-Based Forgery Detection:** Analyzes documents to identify potential signs of forgery.
- **Multiple Document Categories:** Supports PAN cards, Aadhaar cards, marksheets, educational certificates, college IDs, birth certificates, income certificates, experience certificates, medical certificates, and domicile certificates.
- **Document Category Verification:** Allows users to select a document category before analysis.
- **Document Analysis:** Examines uploaded documents using the project's implemented analysis techniques.
- **Report Generation:** Generates analysis reports using the report generation module.
- **User-Friendly Interface:** Provides a simple web interface built using Flask templates and static assets.
- **Signature Analysis:** Includes a dedicated signature analysis module.

## 🛠️ Technologies Used

- Python
- Flask
- TensorFlow / Keras
- HTML, CSS, JavaScript
- Machine Learning and Deep Learning
- SQLite or the database configured in the project

## 📂 Project Structure

```text
AuthentiScan/
├── app.py
├── analysis.py
├── database.py
├── report_generator.py
├── signature_analysis.py
├── requirements.txt
├── models/
│   └── forgery_model.keras
├── static/
├── templates/
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ayuu07-create/AuthentiScan.git
cd AuthentiScan
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

Open the local URL displayed in your terminal, usually `http://127.0.0.1:5000/`.

## 🔍 How It Works

1. Open the AuthentiScan web application.
2. Select the appropriate document category.
3. Upload the document for analysis.
4. Allow the system to process the uploaded document.
5. Review the analysis result and generated report, if available.

## 🎯 Project Objective

The objective of AuthentiScan is to explore how artificial intelligence and document analysis can help identify potentially forged digital documents. 
It aims to provide a convenient initial screening tool for documents across multiple categories.

## ⚠️ Disclaimer
AuthentiScan is an academic project intended for educational and research purposes. Its results are not a guarantee of document authenticity and should not be treated as official verification.
Important documents should be verified through the relevant issuing authority.

## 👨‍💻 Author

Ayuu07-create
GitHub: https://github.com/Ayuu07-create
