from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import fitz
import io
import requests as req

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)


GROQ_API_KEY = "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"  
GROQ_MODEL   = "llama-3.3-70b-versatile"

model         = pickle.load(open("disease_model.pkl", "rb"))
symptoms_list = pickle.load(open("symptoms_list.pkl", "rb"))
medicine_df   = pd.read_csv("disease_medicine.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")
description_df= pd.read_csv("symptom_Description.csv")
severity_df   = pd.read_csv("Symptom-severity.csv")

def get_medicine(disease):
    row = medicine_df[medicine_df["Disease"] == disease]
    return row["Medicine"].values[0] if not row.empty else "Consult Doctor"

def get_precaution(disease):
    row = precaution_df[precaution_df["Disease"] == disease]
    if not row.empty:
        return [p for p in list(row.values[0][1:]) if isinstance(p, str) and p.strip()]
    return ["Consult a healthcare professional"]

def get_description(disease):
    row = description_df[description_df["Disease"] == disease]
    return row["Description"].values[0] if not row.empty else ""

def predict_disease(symptoms):
    input_vector = [0] * len(symptoms_list)
    matched = []
    for symptom in symptoms:
        s = symptom.strip().lower().replace(" ", "_")
        if s in symptoms_list:
            input_vector[symptoms_list.index(s)] = 1
            matched.append(s)
    input_df = pd.DataFrame([input_vector], columns=symptoms_list)
    probs    = model.predict_proba(input_df)[0]
    top3     = np.argsort(probs)[-3:]
    results  = []
    for i in reversed(top3):
        disease = model.classes_[i]
        results.append({
            "disease":     disease,
            "probability": float(round(probs[i] * 100, 2)),
            "medicine":    get_medicine(disease),
            "precaution":  get_precaution(disease),
            "description": get_description(disease),
        })
    return results, matched

def extract_text(file):
    file_bytes = file.read()
    if file.filename.lower().endswith(".pdf"):
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
    else:
        img  = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
    return text

MEDICINE_KB = [
    {"name":"Paracetamol","dosage":"500 mg","timing":"After food – Morning & Night","used_for":"Fever, mild pain","keywords":["paracetamol","dolo","crocin","acetaminophen"]},
    {"name":"Amoxicillin","dosage":"250 mg","timing":"Before food – Twice daily","used_for":"Bacterial infections","keywords":["amoxicillin","mox","amoxil"]},
    {"name":"Ibuprofen","dosage":"400 mg","timing":"After food – Thrice daily","used_for":"Pain, inflammation","keywords":["ibuprofen","brufen","advil","combiflam"]},
    {"name":"Vitamin D3","dosage":"60 000 IU","timing":"Once a week","used_for":"Vitamin D deficiency","keywords":["vitamin d","calcirol","d3","cholecalciferol"]},
    {"name":"Metformin","dosage":"500 mg","timing":"After food – Twice daily","used_for":"Type 2 Diabetes","keywords":["metformin","glycomet","glucophage"]},
    {"name":"Azithromycin","dosage":"500 mg","timing":"Once daily – 3 days","used_for":"Respiratory infections","keywords":["azithromycin","azithral","zithromax"]},
    {"name":"Omeprazole","dosage":"20 mg","timing":"Before food – Once daily","used_for":"Acidity, GERD","keywords":["omeprazole","omez","prilosec"]},
    {"name":"Cetirizine","dosage":"10 mg","timing":"At bedtime","used_for":"Allergies, cold symptoms","keywords":["cetirizine","cetzine","zyrtec"]},
]

def analyse_text(text):
    text = text.lower()
    found, seen = [], set()
    for item in MEDICINE_KB:
        if any(kw in text for kw in item["keywords"]) and item["name"] not in seen:
            found.append({k: item[k] for k in ("name","dosage","timing","used_for")})
            seen.add(item["name"])
    return found

@app.route("/", methods=["GET","POST"])
def home():
    prediction, matched = None, []
    if request.method == "POST":
        user_input = request.form.get("symptoms","")
        if user_input.strip():
            prediction, matched = predict_disease(user_input.split(","))
    return render_template("index.html", prediction=prediction, matched=matched)

@app.route("/report", methods=["GET","POST"])
def report():
    analysis = None
    if request.method == "POST":
        files = request.files.getlist("report")
        all_meds, seen = [], set()
        for f in files:
            if f.filename == "": continue
            try:
                for med in analyse_text(extract_text(f)):
                    if med["name"] not in seen:
                        all_meds.append(med); seen.add(med["name"])
            except Exception as e:
                print(f"Error: {e}")
        analysis = all_meds if all_meds else [{"name":"No medicines detected","dosage":"—","timing":"—","used_for":"Please consult your doctor"}]
    return render_template("report.html", analysis=analysis)

@app.route("/medicine", methods=["GET","POST"])
def medicine():
    analysis = None
    if request.method == "POST":
        files = request.files.getlist("medicine_file")
        all_meds, seen = [], set()
        for f in files:
            if f.filename == "": continue
            try:
                for med in analyse_text(extract_text(f)):
                    if med["name"] not in seen:
                        all_meds.append(med); seen.add(med["name"])
            except Exception as e:
                print(f"Error: {e}")
        analysis = all_meds if all_meds else [{"name":"No medicines detected","dosage":"—","timing":"—","used_for":"Try a clearer image"}]
    return render_template("medicine.html", analysis=analysis)

@app.route("/hospitals")
def hospitals():
    return render_template("hospitals.html")

@app.route("/api/hospitals")
def api_hospitals():
    import math
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    if lat is None or lng is None:
        return jsonify({"error": "lat and lng required"}), 400
    radius = 10000
    overpass_query = f"""
[out:json][timeout:25];
(
  node["amenity"="hospital"](around:{radius},{lat},{lng});
  way["amenity"="hospital"](around:{radius},{lat},{lng});
  node["amenity"="clinic"](around:{radius},{lat},{lng});
  node["healthcare"="hospital"](around:{radius},{lat},{lng});
);
out center tags;
"""
    try:
        resp = req.post("https://overpass-api.de/api/interpreter", data={"data": overpass_query}, timeout=20)
        resp.raise_for_status()
        osm = resp.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    hospitals_list = []
    for el in osm.get("elements", []):
        tags  = el.get("tags", {})
        h_lat = el.get("lat") or el.get("center", {}).get("lat")
        h_lng = el.get("lon") or el.get("center", {}).get("lon")
        if not h_lat or not h_lng: continue
        hosp_name  = tags.get("name") or tags.get("name:en") or "Unknown Hospital"
        dist_km    = haversine(lat, lng, h_lat, h_lng)
        addr_parts = [tags.get("addr:housename",""), tags.get("addr:street",""), tags.get("addr:suburb","") or tags.get("addr:city",""), tags.get("addr:postcode","")]
        address    = ", ".join(p for p in addr_parts if p) or "Address not available"
        phone      = tags.get("phone") or tags.get("contact:phone") or ""
        emergency  = tags.get("emergency") == "yes"
        amb_numbers = [phone] if phone else ["108"]
        hospitals_list.append({"name":hosp_name,"address":address,"distance":f"{dist_km:.1f} km","distance_km":dist_km,"type":tags.get("operator:type","").capitalize() or "Hospital","speciality":tags.get("healthcare:speciality","General").title(),"phone":phone,"ambulance_numbers":amb_numbers,"emergency":emergency,"lat":h_lat,"lng":h_lng})

    hospitals_list.sort(key=lambda x: x["distance_km"])
    return jsonify(hospitals_list[:10])

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data             = request.get_json()
    messages         = data.get("messages", [])
    language         = data.get("language", "English")
    lang_instruction = data.get("langInstruction", "Respond in English.")

    system_prompt = f"""You are MediSense AI, a knowledgeable and compassionate medical health assistant.

CRITICAL LANGUAGE RULE: {lang_instruction}
You MUST respond entirely in {language}. Every word of your reply must be in {language}. Do not switch to English mid-response.

You help patients understand health conditions, symptoms, medicines, nutrition, mental health, and wellness.

Your personality: warm, empathetic, clear, and reassuring. You make patients feel heard.

Guidelines:
- Use simple, friendly language in {language} — avoid heavy jargon
- For life-threatening symptoms (chest pain, difficulty breathing, stroke signs, loss of consciousness, severe bleeding), IMMEDIATELY urge calling 112 or 108 (ambulance)
- Always add a brief note that you are an AI and cannot replace professional medical advice
- Topics: symptoms, diseases, medicines, nutrition, diet, mental health, fitness, preventive care, first aid, general wellness, lab reports, medication timing
- Do NOT prescribe specific dosages — refer to a pharmacist or doctor
- Format: clear paragraphs, bullet points for lists. Mobile-readable.
- If distressed or anxious, acknowledge feelings with empathy first
- You are part of MediSense AI which also has: Symptom Checker, Prescription Analyser, Medicine Analyser, and Hospital Finder

Always end serious symptom discussions with the equivalent of "⚠️ Please consult a doctor for an accurate diagnosis." in {language}."""

    groq_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        resp = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": groq_messages, "max_tokens": 1024, "temperature": 0.7},
            timeout=30,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except req.exceptions.HTTPError as e:
        return jsonify({"error": f"Groq API error: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
