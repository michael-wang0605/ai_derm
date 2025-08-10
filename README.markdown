# **SDERM AI**
SDERM AI (**Skin Disease Early Recognition Model**) is an AI-powered dermatology image classification tool designed to enhance accessibility, accuracy, and inclusivity in skin disease diagnostics, particularly for underprivileged and remote communities.  
This project uses a **pretrained ConvNeXt-Tiny** model fine-tuned on a curated dermatology dataset, enabling it to classify various skin conditions from medical images.

---

## **Features**
- **Pretrained ConvNeXt-Tiny Model** with fine-tuning for higher accuracy.
- **Diverse Skin Tone Support** to reduce bias in diagnostic predictions.
- **End-to-End Pipeline** from training to web-based results.
- Built with **PyTorch**, **FastAPI**, and **TailwindCSS**.

---

## **Dataset**
We used the **DermNet Skin Disease Images Dataset**, available here:  
[https://www.kaggle.com/datasets/umairshahab/dermnet-skin-diesease-images](https://www.kaggle.com/datasets/umairshahab/dermnet-skin-diesease-images)

---

## **Installation & Setup**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/michael-wang0605/ai_derm.git
cd ai_derm
```

### **2️⃣ Install Dependencies**
Ensure you have Python installed along with the dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## **3️⃣ Download and Prepare the Dataset**
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/umairshahab/dermnet-skin-diesease-images).
- Name the folder exactly: `dataset_categorized_final_split`.
- Place it inside the project directory.

## **4️⃣ Train the Model**
Run the following command to train the model and generate `best_model.pth`:

```bash
python base_model.py
```

## **5️⃣ Start the Backend Server**
Run the FastAPI backend with:

```bash
uvicorn server:app --reload
```

## **6️⃣ Launch the Website**
- Open your browser and go to: `http://127.0.0.1:8000`.
- Upload an image to see the prediction results.

## **File Structure**
```
ai_derm/
├── base_model.py                # Training script for best_model.pth
├── server.py                    # Backend API
├── best_model.pth               # Trained model (generated after training)
├── dataset_categorized_final_split/  # Dataset folder (user-provided)
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
```

## **Model Details**
- **Base Model**: ConvNeXt-Tiny (ImageNet pretrained)
- **Fine-tuning**: Entire model trained on DermNet dataset
- **Loss Function**: CrossEntropyLoss with class weighting
- **Optimizer**: Adam / AdamW
- **Mixed Precision Training**: Enabled for speed

## **Credits**
- **Dataset**: DermNet Skin Disease Images
- **Research Guidance**: Prof. Kai Shu, Emory University
- **Development**: Michael Wang

---
