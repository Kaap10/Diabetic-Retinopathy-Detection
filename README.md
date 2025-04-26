
---

## 🗂️ Dataset

- **Messidor Dataset:** Publicly available retinal images for diabetic retinopathy detection.
- **Preprocessing:** Images are resized to 224x224 and normalized.
- **Split:** 80% training, 20% validation (organize as `data/train/<class>` and `data/val/<class>`).

---

## 🚀 Features

- Upload retinal images via a drag-and-drop or file picker interface.
- Real-time prediction of diabetic retinopathy severity.
- Visual and textual feedback, including severity, confidence, description, and recommendations.
- Lightweight, responsive backend for seamless communication.

---

## 💻 Tech Stack

- **Frontend:** React, TypeScript, Tailwind CSS
- **Backend:** Flask, Python
- **Machine Learning:** TensorFlow, Keras
- **Other:** Pillow, NumPy

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone "https://github.com/Kaap10/Diabetic-Retinopathy-Detection.git"

cd Diabetic-Retinopathy-Detection
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

- Download the Messidor dataset (or similar).
- Organize images into `data/train/<class>` and `data/val/<class>` directories.

### 4. Train the Model

```bash
python model/cnn_model.py
```
- This will save the trained model as `model/model.h5`.

### 5. Run the Flask Backend

```bash
python Backend/app.py
```
- The backend will be available at `http://127.0.0.1:5000/`.

### 6. Start the Frontend

- If using Vite/React, run:
  ```bash
  npm install
  npm run dev
  ```
- Open your browser at the provided local address (usually `http://localhost:5173/`).

---

## 🖼️ Usage

1. Open the web app in your browser.
2. Upload a clear, high-quality retinal image (JPEG/PNG, max 5MB).
3. Wait for the AI to analyze the image.
4. View the predicted severity, confidence, description, and recommendations.
5. Note: This tool is for educational/demo purposes and does not replace professional medical advice.

---

## 🌟 Future Enhancements

- Improve model accuracy with advanced architectures and more data.
- Add more sophisticated image preprocessing.
- Batch image support and result history.
- Deploy the app to a cloud platform for public access.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or suggestions, open an issue or contact via GitHub.
