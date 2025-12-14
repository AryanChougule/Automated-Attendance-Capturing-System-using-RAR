🚀 RAR-Based Automated Attendance System

An AI-powered attendance system built using Retrieval-Augmented Recognition (RAR), MTCNN for face detection, and ArcFace for vector-based face verification.
This system captures frames, detects faces, converts them into embeddings, retrieves the closest match from a stored database, and automatically records attendance in a CSV file.

📌 Project Overview

Unlike traditional CNN-based classifiers, this system uses a retrieval-based recognition pipeline, making it:

⚡ Fast

📈 Scalable

🎯 Accurate

🔍 Explainable

🧩 Easy to update (no retraining required)

Using ArcFace embeddings and cosine similarity, the system identifies individuals based on vector proximity — similar to how RAG works in NLP, but adapted for computer vision, hence RAR (Retrieval-Augmented Recognition).

🧠 How It Works

Face Detection:
MTCNN detects faces from a frame or input image.

Embedding Generation:
Detected faces are passed through ArcFace to generate a 512-dimensional vector.

Retrieval-Augmented Recognition (RAR):
Embeddings are compared with stored student embeddings using cosine similarity.

Attendance Marking:
If similarity > threshold → identity confirmed → attendance logged into a CSV file.

CSV Output:
Each row contains:

Student ID

Status (Present/Absent)

Timestamp

📁 Project Structure
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


(You can adjust this structure as per your actual project.)

🛠 Technologies Used

MTCNN – Face detection

ArcFace (ONNX) – Embedding generation

Cosine Similarity – Identity verification

RAR (Retrieval-Augmented Recognition) – Matching logic

Python, NumPy, Pandas, OpenCV

CSV Logging

📖 How to Implement This For Your Use

Follow these steps to run and customize the system for your own attendance workflow.

1️⃣ Download or Clone the Repository
git clone <your-repo-link>
cd <your-project-folder>


(Replace <your-repo-link> with your GitHub URL.)

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Prepare Your Dataset

Inside the raw_dataset/ folder, create subfolders:

raw_dataset/
│── 101/
│── 102/
│── 103/
│── ...


Each folder should contain 15–20 images of one person.
You can use your phone or laptop camera.

4️⃣ Generate Embeddings

Run the script to convert raw images into ArcFace vectors:

python generate_embeddings.py


This will produce:

embeddings.pkl


which contains the final identity vectors used for recognition.

5️⃣ Run the Attendance System
python main.py


The CLI will show:

live frame extraction

face detection

embedding comparison

recognized student ID

attendance CSV update

6️⃣ Check the Output CSV

Generated file example:

attendance.csv
StudentID,Status,Timestamp
101,Present,2025-01-12 09:13:27
102,Present,2025-01-12 09:14:03
...


This CSV can be used for:

dashboards

web applications

analytics

reporting

✏️ Sections for You to Edit

(You can fill these in after finalizing your project.)

📌 Motivation

Write why you built this project…

📌 Team Members

Add the names of your teammates…

📌 Under Guidance Of

Add your mentor’s name (Apurva Ma’am)…

📌 Demo Video / Screenshots

Add GIFs, images, YouTube video…

📌 Future Improvements

Describe what you plan to add next…

🎯 Conclusion

This project demonstrates how Retrieval-Augmented Recognition (RAR) can be applied to face identification tasks like attendance management.
By combining MTCNN, ArcFace, and vector similarity search, the system achieves high accuracy, scalability, and real-world usability.
