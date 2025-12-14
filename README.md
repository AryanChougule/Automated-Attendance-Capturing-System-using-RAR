# 🚀 RAR-Based Automated Attendance System

An AI-powered attendance system built using **Retrieval-Augmented Recognition (RAR)**, **MTCNN** for face detection, and **ArcFace** for vector-based face verification.  
Instead of traditional CNN-based classification, this system uses **vector similarity retrieval**, making the process faster, more scalable, and more accurate.

---
# IF you want to implement this, below I have provided each and every step with explanation. A new beginner can also understand the flow just read it slowly and patiently.

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
│── embeddings.pkl or student_embeddings.pkl in my repsotiory   # stored ArcFace embeddings (auto-generated)
│── working/attendance.csv       # attendance logs
|── camera_test.py and camera_test2.py  # Checks the camera is working ok.
│── temp_faces/    # Stores face captured from the MTCNN for visual representation
│── README.md
|── face_embd.py  # run this file once, converts the raw dataset to embeddings dataset
│── requirements.txt # create a python or conda virtual environment and then install the requirements 
└── main.py   # after all steup run this file in the virtual environment that has the dependences installed
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
git clone https://github.com/AryanChougule/Automated-Attendance-Capturing-System-using-RAR
cd <your-project-folder>
```

---

## 2️⃣ Install Dependencies
First create a virtual environment and install the dependences (This dependences are for python version 3.10) See if they work for your python version. 
```bash
pip install -r requirements.txt
```

---

## 3️⃣ Prepare Your Raw Dataset

Inside `raw_dataset/`, create subfolders named by student IDs:

```
raw_dataset/
│── 101/images of that student
│── 102/
│── 103/
│── ...
```

Each folder should contain **15–20 images** of the same person.

---

## 4️⃣ Generate Embeddings

Run the script to convert raw images into ArcFace embeddings :

```bash
python face_embd.py
```

This will create:

```
embeddings.pkl or student_embeddings.pkl
```

---
## Extra Step
   
   Like to add one more step is you are using a different camera for frame capturing
   If you want to use mobile camera then download app "IP webcam" from playstore and copy IP (provided in the app ) paste the IP in the main.py as I have stated in the main.py file.
   And run the camera_test.py and camera_test2.py file to see the working

   
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
You will see this file in the working folder
Example:

```
attendance.csv
StudentID,Status,Timestamp
101,Present,2025-01-12 09:13:27
102,Present,2025-01-12 09:14:03
```

This file can be used for web applications, dashboards, or reporting.

---

## 📌 Motivation
Want to try a different approach than tradition CNN and also, where a CNN require 100s of images per student for training, our system just require 10-15 images per student and also dont require model training so easily scalable.

## 📌 Team Members
- **Aryan Chougule** (Myself) — [LinkedIn Profile](https://www.linkedin.com/in/aryanpravinchougule/)
- **Shreejeet Gaikwad**
- **Varun Kulkarni**
- **Atharv Halwai**
  Contact any one of use for any problem
## 📌 Demo Video / Screenshots


## 📌 Achivements
Got seed funded from Rajarambapu Institute of Technology, Islampur to implement this system in classrooms.

---

# 🎯 Conclusion

This project demonstrates how **Retrieval-Augmented Recognition** can be applied to real-world face identification and attendance automation.  
By combining MTCNN, ArcFace, and vector similarity search, the system achieves high accuracy, scalability, and robustness.

---
