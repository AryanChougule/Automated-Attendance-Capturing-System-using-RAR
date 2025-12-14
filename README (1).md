# 🚀 RAR-Based Automated Attendance System

An AI-powered attendance system built using **Retrieval-Augmented Recognition (RAR)**, **MTCNN** for face detection, and **ArcFace** for vector-based face verification.  
Instead of traditional CNN-based classification, this system uses **vector similarity retrieval**, making the process faster, more scalable, and more accurate.

---

## 📌 Project Overview

This system captures frames, detects faces, generates embeddings, retrieves the closest match from a stored database, and automatically records attendance in a CSV file.

Using the concept of **RAR (Retrieval-Augmented Recognition)**, the system identifies individuals based on vector proximity—similar to RAG in NLP but adapted for computer vision.

---

## 🧠 How It Works

1. **Face Detection (MTCNN)**  
   Detects faces from images or video frames.

2. **Embedding Generation (ArcFace)**  
   Converts each face into a **512-dimensional vector**.

3. **Retrieval-Augmented Recognition (RAR)**  
   Compares embeddings with stored vectors using cosine similarity.

4. **Attendance Marking**  
   If similarity exceeds a threshold → identity confirmed → attendance added to CSV.

5. **CSV Output**  
   Contains student ID, status, and timestamp.

---

## 📁 Project Structure

```
your-project/
│── raw_dataset/         # student images (15–20 images per person)
│── embeddings.pkl       # stored ArcFace embeddings (auto-generated)
│── attendance.csv       # attendance logs
│── src/
│    ├── detect.py       # MTCNN detection pipeline
│    ├── recognize.py    # ArcFace recognition + RAR logic
│    ├── utils.py        # helper functions
│── README.md
│── requirements.txt
└── main.py
```

---

## 🛠 Technologies Used

- **MTCNN** – Face detection  
- **ArcFace (ONNX)** – Embedding generation  
- **Cosine Similarity** – Identity matching  
- **RAR** – Retrieval-Augmented Recognition  
- **Python, NumPy, Pandas, OpenCV**  
- **CSV Logging**

---

# 📖 How to Implement This For Your Use

Follow these steps to set up and run the system for your own use case.

---

## 1️⃣ Download or Clone the Repository

```bash
git clone <your-repo-link>
cd <your-project-folder>
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Prepare Your Raw Dataset

Inside `raw_dataset/`, create subfolders named by student IDs:

```
raw_dataset/
│── 101/
│── 102/
│── 103/
│── ...
```

Each folder should contain **15–20 images** of the same person.

---

## 4️⃣ Generate Embeddings

Run the script to convert raw images into ArcFace embeddings:

```bash
python generate_embeddings.py
```

This will create:

```
embeddings.pkl
```

---

## 5️⃣ Run the Attendance System

```bash
python main.py
```

You will see:

- Frame capturing  
- Face detection  
- Embedding comparison  
- Identity retrieval  
- Attendance updates  

---

## 6️⃣ Output CSV File

Example:

```
attendance.csv
StudentID,Status,Timestamp
101,Present,2025-01-12 09:13:27
102,Present,2025-01-12 09:14:03
```

This file can be used for web applications, dashboards, or reporting.

---

# ✏️ Sections for You to Edit

## 📌 Motivation
_Add your project motivation here..._

## 📌 Team Members
_Add your teammates' names here..._

## 📌 Under Guidance Of
_Add mentor’s name here..._

## 📌 Demo Video / Screenshots
_Add project images or links..._

## 📌 Future Improvements
_Add your planned upgrades..._

---

# 🎯 Conclusion

This project demonstrates how **Retrieval-Augmented Recognition** can be applied to real-world face identification and attendance automation.  
By combining MTCNN, ArcFace, and vector similarity search, the system achieves high accuracy, scalability, and robustness.

---
