# Algorithmic Justice League (AJL) Kaggle Competition: AI for Equitable Dermatology
## Spring 2025 AI Studio
---

### **👥 Team Members**

Aisha Malik, Paulina Calderón (Ana), Mysara Elsayed, Rishita Dhalbisoi, Yousra Awad, Zohreh Ashtarilarki

---

## **🎯 Project Highlights**

* Developed an ensemble model using transfer learning and fine-tuning of multiple models for image classification of skin diseases.
* Achieved a ranking of **2nd place virtual team** and **5th place on the Kaggle leaderboard**.
* Implemented data augmentation and image preprocessing techniques to mitigate class imbalance and enhance model performance.

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏾‍💻 Setup & Execution**

1. **Cloning the Repository**
To clone the repository to your local machine, use the following Git command:
```bash
git clone https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology.git
```

2. **Accessing The Datasets**
The dataset for this competition can be located here: [Kaggle Dataset](https://www.kaggle.com/competitions/bttai-ajl-2025/data). Navigate to the **Download** button and extract it into a folder in your repository directory.

3. **Running the Python Scripts**  
   To run the Python scripts, ensure the environment is set up and dependencies are installed. Then, execute the scripts from the command line as follows:
   ```bash
   python your_script.py
   ```
   Replace `your_script.py` with the specific Python file you wish to run. The scripts will process the data and train the models as per the instructions within the code.

---

## **🏗️ Project Overview**

This competition, part of the Break Through Tech AI program in collaboration with the **Algorithmic Justice League**, focuses on a critical issue in dermatology AI: **underperformance for darker skin tones**.

Our goal was to train a machine learning model capable of classifying **21 different skin conditions** across various skin tones, ensuring fairness and accuracy in medical AI applications.

---

## **📊 Data Exploration**

### **Dataset Overview:**
* Utilized the official **Kaggle AJL dataset**, a subset of the **FitzPatrick17 dataset** (~17,000 dermatology images labeled with skin conditions and Fitzpatrick skin types).
* Investigated **class imbalance** across **skin condition labels** and **skin tone categories**.
* The dataset consists of approximately **4500 images** representing **21 skin conditions** from the **FitzPatrick17k dataset**, which contains over 100 dermatological conditions.
* Images are sourced from two reputable dermatology websites: **DermaAmin** and **Atlas Dermatologico**.
* The dataset contains a **range of skin tones** based on the **Fitzpatrick skin type (FST)** scale (from **1 to 6**), with **Skin Tone 2** being the most represented and **Skin Tones 5 and 6** being underrepresented.

### **Key Data Insights:**

#### **Visualizations:**

1. **Skin Condition Distribution (Before Augmentation)**  
   This chart shows the distribution of different skin conditions in the dataset before data augmentation. There is a noticeable class imbalance, with conditions like **squamous-cell-carcinoma**, **basal-cell-carcinoma**, and **folliculitis** being more prevalent, while conditions such as **seborrheic-keratosis** and **basal-cell-carcinoma-morpheiform** are underrepresented.

   ![Skin Condition Distribution (Before Augmentation)](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Skin%20Condition%20Distribution%20(Before%20Augmentation).png)


2. **Skin Tone Distribution (Before Augmentation)**  
   The dataset is also imbalanced across skin tones, with **Skin Tone 2** (Fitzpatrick scale) being the most represented, followed by **Skin Tone 3** and **Skin Tone 1**. Skin tones **5** and **6** are significantly underrepresented.

   ![Skin Tone Distribution (Before Augmentation)](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Skin%20Tone%20Distribution%20(Before%20Augmentation).png)


3. **Skin Condition Distribution (After Augmentation)**  
   After applying data augmentation, the classes were balanced. Augmentation techniques like flipping, rotation, and brightness/contrast adjustments were used to create a more even distribution across the different conditions, allowing for better model training.

   ![Skin Condition Distribution (After Augmentation)](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Skin%20Condition%20Distribution%20(After%20Augmentation).png)


4. **Skin Tone Distribution (After Augmentation)**  
   The augmentation process also addressed the skin tone imbalance. All skin tones, especially **Skin Tone 1, 2, 3**, and **5**, were balanced to have approximately equal representation in the dataset, ensuring fairer model predictions across all skin tones.

   ![Skin Tone Distribution (After Augmentation)](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Skin%20Tone%20Distribution%20(After%20Augmentation).png)

#### **Preprocessing Techniques:**
* **Data Augmentation:** Techniques like flipping, rotation, and brightness/contrast adjustments were used to balance the skin tone representation across the dataset. This helped mitigate the underrepresentation of darker skin tones and rare skin conditions.
* **Normalization & Resizing:** Image sizes were standardized to consistent dimensions for model training. This included resizing all images to a fixed size to ensure compatibility across different models.

---

## **🧠 Model Development**

### **🧬 Models Used:**
We experimented with and fine-tuned various deep learning architectures:
* **ResNet50**
* **MobileNet**
* **NASNet**
* **EfficientNetB3 & EfficientNetB4**
* **DenseNet121**

### **🌀 Training Strategy:**
* Used **transfer learning** and **fine-tuned** models on the AJL dataset.
* **Evaluation Metric:** Weighted **F1-score** (ensuring balanced performance across classes).

### **⚙️ Ensemble Learning:**
* We built and trained each model **separately**, saving their trained states after a certain number of epochs.
* Later, we **combined the models** using both:
  - **Hard Voting:** Selected the most frequently predicted label across models.
  - **Soft Voting (best results):** Averaged model confidence scores and chose the highest probability class.
* The **best performing ensemble** consisted of:
  - **ResNet50 + EfficientNetB3 + EfficientNetB4 + DenseNet121**

Model Performance:
Below is the accuracy and loss curve of a single model (DenseNet121) during initial training and fine-tuning, which illustrates the improvement in performance across epochs.

   ![Accuracy_Loss_DenseNet121_Fine_tuning](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/desnsenet121_history.png)

---

### **📈 Results & Key Findings**

### **📊 Performance Metrics:**
* **Final Kaggle Leaderboard Score:** Fluctuated between **3rd and 4th place** in final submissions.
* **Best Model Performance:** Ensemble of ResNet50, EfficientNetB3, EfficientNetB4, and DenseNet121 using soft voting.
* **Observations:**
  - Ensembling significantly improved accuracy and fairness.
  - MobileNet and NASNet performed **worse** than expected and were excluded from the final model.

---

## **Confusion Matrix Analysis of the Ensemble Model (Soft Voting)**

The confusion matrix for the ensemble model, which uses soft voting, provides valuable insights into its performance across different classes. Below are the key takeaways:

### **Class Confusion:**

- **Basal-cell-carcinoma-morpheiform** and **malignant-melanoma** are more frequently misclassified, as observed from the higher off-diagonal values in these rows and columns. This suggests that the model might struggle to differentiate between these particular classes.
  
- **Prurigo-nodularis**, **Dyshidrotic-eczema**, and **Dermatomyositis** show relatively higher accuracy, with fewer misclassifications, highlighting the model's better ability to identify these classes correctly.

### **Accuracy and Misclassification Patterns:**

- Overall, the model performs reasonably well in classifying some of the classes. However, the most frequent misclassifications occur between similar-looking or more challenging classes, as seen in the higher values in off-diagonal elements. This is typical in many multi-class classification tasks, especially with similar categories.

- The diagonal values of the matrix show the number of correct predictions for each class. The overall diagonal sum indicates a decent level of accuracy in the classification process, though the model could still be refined to reduce misclassifications, especially for harder-to-classify categories.

### **Implications for Model Improvement:**

- The soft voting method seems to have been effective in aggregating predictions from multiple models, boosting the overall performance. However, focusing on the more challenging classes (**Basal-cell-carcinoma-morpheiform** and **malignant-melanoma**) and further refining the model through techniques like class weighting or additional fine-tuning could further enhance performance.

- Visualizing the confusion matrix can guide efforts in identifying patterns of misclassification and can be used to focus on areas where the model struggles the most.

### **Confusion Matrix:**

The following is the confusion matrix showing how the ensemble model performed across different categories:

   ![Confusion_Matrix_ensemble_model](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Confusion%20Matrix.png)
   
---

## **🎨 Impact Narrative**

### **AJL Challenge:**
Our approach tackled dermatology AI bias by:
* **Applying data augmentation** to balance training samples across skin tones.
* **Evaluating fairness** through separate validation metrics for different Fitzpatrick skin types.
* **Promoting equitable AI practices** that could mitigate disparities in medical diagnosis for underrepresented skin tones.

This work highlights **ethical concerns in healthcare AI** and the need for **inclusive datasets** in dermatology models.

---

## **🚀 Next Steps & Future Improvements**

### **🧠 Future Directions:**
* **Alternative Model Architectures:**
  - Explore **custom CNNs** instead of pre-trained ImageNet models.
  - Implement models designed **from scratch** for dermatology classification.

* **Additional Techniques:**
  - **Use external datasets** to improve skin tone representation.
  - **Apply GAN-based augmentation** to generate synthetic images and balance underrepresented classes.
  - **Investigate attention-based models** (e.g., Vision Transformers) for improved feature extraction.

---

## **📜 References & Additional Resources**

* ["What is Fairness?" by Berkeley Haas](https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf)
* [TensorFlow Documentation](https://www.tensorflow.org)
* [Kaggle Tutorials on Skin Classification](https://www.kaggle.com/code/smitisinghal/skin-disease-classification)
* [Fine-tuning and Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
* [An Integrated Deep Learning Model with EfficientNet and ResNet for Accurate Multi-Class Skin Disease Classification](https://www.mdpi.com/2075-4418/15/5/551)


---
