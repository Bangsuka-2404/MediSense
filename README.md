🩺 MediSense AI – Intelligent Healthcare Assistant

MediSense AI is an AI-powered full-stack healthcare web application designed to provide instant preliminary medical insights using Machine Learning, OCR, and Large Language Models. The platform helps users predict possible diseases based on symptoms, analyze prescriptions, detect medicines from medical reports, locate nearby hospitals, and interact with an AI-powered medical assistant in multiple languages.

🚀 Features

🤖 AI Medical Chat Assistant
Real-time health guidance powered by Llama 3 via Groq API

🧠 Symptom-Based Disease Prediction
Predicts possible diseases based on user-entered symptoms using ML model

📄 Prescription Analysis (OCR)
Upload medical prescriptions to extract medicine details automatically

💊 Medicine Detection from Reports
Detects medicine name, dosage, timing, and usage from uploaded images or PDFs

🚑 Nearby Hospital Locator
Uses geolocation and OpenStreetMap API to display nearby hospitals

🌍 Multi-language Support
Supports regional languages for accessibility

📊 Machine Learning Integration
Uses trained ML model for disease prediction

🔐 Secure API Integration
API keys stored securely using environment variables

🛠️ Tech Stack
Frontend

HTML5

CSS3

JavaScript

Leaflet.js (Maps)

Backend

Python

Flask

AI / ML

Scikit-learn

Groq LLM API (Llama 3)

OCR & File Processing

Tesseract OCR

PyMuPDF (fitz)

Pillow

APIs

Groq API

OpenStreetMap Overpass API


⚙️ Installation & Setup (Local)
1️⃣ Clone Repository
git clone https://github.com/yourusername/MediSense-AI.git
cd MediSense-AI

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install Tesseract OCR
Windows:

Download from:
https://github.com/tesseract-ocr/tesseract

Linux (Ubuntu):
sudo apt update
sudo apt install tesseract-ocr
4️⃣ Set Environment Variable

Create a .env file or set manually:

GROQ_API_KEY=your_groq_api_key_here
5️⃣ Run Application
python app.py

Visit:

http://127.0.0.1:5000
☁️ Deployment on Render
1️⃣ Create requirements.txt
Flask
gunicorn
numpy
pandas
pillow
pytesseract
pymupdf
requests
scikit-learn


👨‍💻 Author

Developed by: Bangsuka Haldar

📜 License

This project is for educational and research purposes only.
Not intended to replace professional medical advice.
