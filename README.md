# Algorithmic Justice League (AJL) Kaggle Competition: Equitable AI for Dermatology
## Spring 2025 AI Studio
---

### **👥 Team Members**

Aisha Malik, Paulina Calderón (Ana), Mysara Elsayed, Rishita Dhalbisoi, Yousra Awad, Zohreh Ashtarilarki

---

## **🎯 Project Highlights**

* Developed an ensemble model using transfer learning and fine-tuning of multiple models for image classification of skin diseases.
* Achieved a ranking of **2nd place virtual team** and **4th place overall** on the Kaggle leaderboard.
* Secured a **0.72415 accuracy** on the public leaderboard (2nd place virtual team and 4th place overall).
* Achieved a **0.68409 accuracy** on the private leaderboard, securing **5th place**.
* Implemented data augmentation and image preprocessing techniques to mitigate class imbalance and enhance model performance.

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏾‍💻 Setup & Execution**

1. **Cloning the Repository**
To clone the repository to your local machine, use the following Git command:
```bash
git clone https://github.com/aisha1021/AJL_Equitable-AI-for-Dermatology.git
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
   The dataset is also imbalanced across skin tones, with **Skin Tone 2** (Fitzpatrick scale) being the most represented, followed by **Skin Tone 3** and **Skin Tone 1**. Skin tones **-1**, **5** and **6** are significantly underrepresented.

   ![Skin Tone Distribution (Before Augmentation)](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Skin%20Tone%20Distribution%20(Before%20Augmentation).png)


3. **Skin Condition Distribution (After Augmentation)**  
   After applying data augmentation, the classes were balanced. Augmentation techniques like flipping, rotation, and brightness/contrast adjustments were used to create a more even distribution across the different conditions, allowing for better model training.

   ![Skin Condition Distribution (After Augmentation)](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Skin%20Condition%20Distribution%20(After%20Augmentation).png)


4. **Skin Tone Distribution (After Augmentation)**  
   The augmentation process also addressed the skin tone imbalance. All skin tones, especially **Skin Tone 1, 2, 3**, and **5**, were balanced to have approximately equal representation in the dataset, ensuring fairer model predictions across all skin tones.

   ![Skin Tone Distribution (After Augmentation)](https://github.com/aisha1021/AJL_AI-for-Equitable-Dermatology/blob/c6678f30bc76f740b1d6c91fb24f8acbc5f51133/images/Skin%20Tone%20Distribution%20(After%20Augmentation).png)

#### **Preprocessing Techniques:**

- **Data Augmentation:**
  To address the class imbalance and improve model generalization, various data augmentation techniques were employed. These techniques aimed to create a more balanced representation of skin tones and skin conditions across the dataset, particularly focusing on underrepresented categories. 
  - **Flipping:** Images were randomly flipped horizontally to introduce diversity and simulate different viewing angles.
  - **Rotation:** Images were randomly rotated by up to 30 degrees to help the model generalize over various orientations of skin conditions.
  - **Shifting:** Random shifts in both horizontal and vertical directions (up to 20% of the image width/height) were applied to simulate different positions of the subject.
  - **Zooming & Shearing:** These transformations help the model learn to handle images with varying scales and slight distortions.
  - **Brightness/Contrast Adjustments:** Random modifications to brightness and contrast were applied to simulate different lighting conditions, ensuring the model can handle diverse image qualities.

  This augmentation not only mitigated the underrepresentation of darker skin tones but also addressed rare skin conditions, providing a more diverse set of training data.

- **Normalization & Resizing:**
  To ensure consistent input to the neural network, image preprocessing involved the following steps:
  - **Normalization:** All image pixel values were scaled to a range of 0 to 1 by dividing by 255.0. This step ensures that the model's training process is more stable and improves convergence.
  - **Resizing:** Images were resized to a fixed size of 224x224 pixels, which is the standard input size for many pre-trained models (e.g., DenseNet121). This resizing step ensures compatibility across different models and provides consistent input dimensions.

   This approach ensures that the model is trained with a diverse set of images, improving its ability to generalize across various real-world scenarios. It also helps address any class imbalance or skin tone underrepresentation by providing more samples for minority classes and skin tones.
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
* **Final Kaggle Leaderboard Score:** Our team's final ranking fluctuated between **3rd and 4th place** on the leaderboard during the competition's final submissions, with a private leaderboard accuracy of **0.68409**, securing **5th place**. On the public leaderboard, we achieved an accuracy of **0.72415**, placing **2nd as a virtual team** and **4th overall**.
* **Best Model Performance:** The most successful model was an **ensemble of ResNet50, EfficientNetB3, EfficientNetB4, and DenseNet121**. These models were combined using **soft voting** to leverage the strengths of each architecture and improve overall performance. This ensemble approach not only enhanced accuracy but also helped the model generalize better across various skin tones and conditions.
* **Observations:**
  - **Ensembling Significantly Improved Performance:** By combining multiple models, we saw a noticeable improvement in both accuracy and fairness. The ensemble mitigated the weaknesses of individual models, providing more robust predictions across diverse image types.
  - **Model Exclusions:** Models like **MobileNet** and **NASNet** underperformed relative to expectations, particularly in terms of both accuracy and fairness. Their inability to generalize well across the diverse dataset led to their exclusion from the final ensemble model. Despite their efficiency in other scenarios, these models struggled with the specific nuances of the dermatological image classification task.
  - **Model Calibration:** After training, we fine-tuned the models' hyperparameters and utilized techniques like **learning rate annealing** and **early stopping** to avoid overfitting and ensure that the ensemble performed optimally across both the public and private leaderboards.
  
The ensemble of ResNet50, EfficientNetB3, EfficientNetB4, and DenseNet121 effectively leveraged the strengths of each model, resulting in better overall performance and higher rankings in the competition.

--- 

## **Confusion Matrix Analysis of the Ensemble Model (Soft Voting)**

The confusion matrix for the ensemble model, which uses soft voting, provides valuable insights into its performance across different classes. Below are the key takeaways, including individual model performances and how the ensemble aggregates their strengths and weaknesses.

### **Confusion Matrix:**

The following confusion matrix shows the individual confusion matrices for each model (DenseNet121, EfficientNetB3, EfficientNetB4, and ResNet50) along with the ensemble model's confusion matrix, which aggregates the predictions of all four models across different categories:

![Confusion_Matrix_ensemble_model](https://github.com/aisha1021/AJL_Equitable-AI-for--Dermatology/blob/572a6dcf8959d1316a2c88111520070d19056225/images/models_confusion_matrix.png)

### **Individual Model Performance:**

- **DenseNet121**: This model made minimal misclassifications, only classifying 3 images as **acne-vulgaris** instead of **acne**. This shows that while DenseNet121 is generally effective, it occasionally struggles with these two similar classes.

- **EfficientNetB3**: Similar to DenseNet121, EfficientNetB3 also misclassified 2 images as **acne-vulgaris** instead of **acne**, highlighting a minor weakness in distinguishing between these two categories.

- **EfficientNetB4**: This model performed exceptionally well with **100% accuracy**, making no misclassifications. Its flawless performance highlights its strong capability in handling the dataset.

- **ResNet50**: ResNet50 misclassified 5 images as **acne-vulgaris** instead of **acne**, which reflects a similar issue as observed with DenseNet121 and EfficientNetB3.

### **Class Confusion:**

- **acne** and **acne-vulgaris** are the most frequently misclassified classes, with these images being confused with each other, as indicated by the higher off-diagonal values in the confusion matrix. This suggests that the models, particularly DenseNet121, EfficientNetB3, and ResNet50, struggle to differentiate between these two classes.

- Other classes such as **Prurigo-nodularis**, **Dyshidrotic-eczema**, and **Dermatomyositis** show relatively higher accuracy, with fewer misclassifications, reflecting the models' better ability to correctly identify these categories.

### **Ensemble Model Strengths:**

The ensemble model, using soft voting, takes the strengths and weaknesses of each individual model into account. By aggregating the predictions of DenseNet121, EfficientNetB3, EfficientNetB4, and ResNet50, the ensemble is able to compensate for the individual misclassifications, particularly with **acne** and **acne-vulgaris**, improving overall performance.

- **DenseNet121** and **EfficientNetB3** both misclassified a few images as **acne-vulgaris**, but **EfficientNetB4**, with its perfect accuracy, and **ResNet50**, while making some mistakes, help balance out the ensemble's predictions.

- The ensemble method, by leveraging the diversity in individual models, results in a more robust performance, where the misclassifications of one model can be corrected by others, leading to an overall improvement.

### **Accuracy and Misclassification Patterns:**

- Overall, the ensemble model performs exceptionally well in classifying most of the classes. However, the most frequent misclassifications occur between similar-looking or more challenging classes, particularly **acne** and **acne-vulgaris**. This is common in many multi-class classification tasks, especially with similar categories.

- The diagonal values of the matrix show the number of correct predictions for each class. The overall diagonal sum indicates a strong level of accuracy in the classification process, although there is room for refinement in distinguishing between the more challenging classes.

### **Implications for Model Improvement:**

- The soft voting method has proven effective in aggregating predictions from multiple models, boosting the overall performance. However, focusing on further refining the models' ability to differentiate between classes like **acne** and **acne-vulgaris** could help reduce misclassifications in the future. Techniques like class weighting or additional fine-tuning of individual models could further improve performance.

- Visualizing the confusion matrix is a valuable tool for identifying patterns of misclassification and helps pinpoint the areas where the model struggles the most, guiding future improvements.

---

## **🎨 Impact Narrative**

### **AJL Challenge:**
Our approach to the **AJL Challenge** aimed at addressing the issue of dermatology AI bias by focusing on inclusivity and fairness. Here's how we made an impact:

* **Applying Data Augmentation to Balance Skin Tones:** To tackle the problem of underrepresentation of darker skin tones in medical datasets, we employed **data augmentation** techniques such as rotation, flipping, brightness adjustment, and more. This helped simulate a more diverse set of images, ensuring that our model was exposed to a variety of skin tones during training, and reducing the risk of bias toward lighter skin tones. This balanced dataset helped mitigate the skewed distribution often seen in medical imaging datasets.
  
* **Evaluating Fairness with Separate Validation Metrics for Fitzpatrick Skin Types:** Instead of treating the skin tone as a single homogeneous category, we divided our validation metrics across distinct **Fitzpatrick skin types**. This allowed us to evaluate the model's performance across a broader spectrum of skin tones and ensured that the model was not overly optimized for any specific subset of skin types. This evaluation helped us pinpoint any disparities in performance and fine-tune the model to ensure fairness and accuracy for all skin tones.

* **Promoting Equitable AI Practices:** Our efforts in ensuring that the model performed equitably across all skin tones are directly aligned with the need to mitigate disparities in medical AI. By focusing on **inclusive datasets** and **fairness metrics**, we worked towards building a dermatology model that could potentially reduce the diagnostic gaps seen in current AI systems, where models tend to underperform when faced with images of darker skin tones.

This work shines a spotlight on **ethical concerns in healthcare AI**, where algorithmic biases can have life-altering consequences, particularly in medical diagnosis. Our approach calls for a shift in AI development to ensure that **inclusive datasets** are not just optional, but essential for any AI application in sensitive domains like dermatology.

---

### **Model Error Rate Across Fitzpatrick Skin Types:**
The results from our model error rate analysis across Fitzpatrick skin types underscore the effectiveness of our fairness-focused approach. The bar chart visualizing the error rates for different models shows consistently low error rates across the Fitzpatrick skin types, with the **ensemble model** demonstrating a robust performance. 

- For most skin types, the error rates are near zero, which indicates that the models are achieving high accuracy.
- Notably, the **ResNet50** model shows a slightly higher error rate for Fitzpatrick types 1, 2, and 4, which highlights a small disparity that was addressed during the fine-tuning process. However, this may be due to the original imbalance in the image data, which had a higher number of images from these three skin types, causing the model to be more susceptible to making errors in these categories.
- The **ensemble model** outperforms individual models, further emphasizing the importance of combining multiple models to minimize bias and enhance accuracy across diverse skin types.

  ![model_error_rate_skin_tones](https://github.com/aisha1021/AJL_Equitable-AI-for-Dermatology/blob/a918ea45b9a5b5647b6ebcb2f7686438e058f92a/images/model_error_rate_skin_tones.png)

   These results not only demonstrate the effectiveness of our approach but also affirm the importance of inclusivity in dermatology AI. By focusing on fairness and accuracy for all skin tones, we aim to bridge the diagnostic gaps that exist in current AI systems. This reinforces our commitment to developing AI models that are not just accurate, but also ethical and equitable for all users.
---

## **🚀 Next Steps & Future Improvements**

### **🧠 Future Directions:**

* **Alternative Model Architectures:**
  - **Explore Custom CNNs:** While pre-trained models like ResNet and EfficientNet offer significant benefits due to their generalization, we plan to **design custom convolutional neural networks (CNNs)** tailored specifically for dermatology classification. By leveraging domain-specific architectural choices, we hope to build models that are not only more efficient but also more effective in capturing dermatological features unique to skin diseases.
  - **Implement Models from Scratch:** Another direction is to explore creating **models from scratch**, using deep learning techniques that are fully customized for dermatology. This would allow us to better optimize the architecture for specific dermatological features, potentially enhancing performance in rare and underrepresented conditions.

* **Additional Techniques for Model Enhancement:**
  - **Use External Datasets to Improve Skin Tone Representation:** To further improve our model's fairness and performance, we plan to incorporate **external dermatology datasets** that contain a wider and more representative variety of skin tones. By augmenting our dataset with more diverse images, we can improve the model's generalization capabilities and ensure it performs better across all skin types.
  - **Apply GAN-Based Augmentation for Synthetic Image Generation:** **Generative Adversarial Networks (GANs)** could be used to **generate synthetic images** to augment underrepresented classes. This method has the potential to generate high-quality dermatological images that mirror real-world data, thus improving class balance and providing additional training examples for rare skin conditions. This could be especially useful in addressing imbalances for underrepresented skin conditions and rare diseases.
  - **Investigate Attention-Based Models (e.g., Vision Transformers):** Attention mechanisms, especially **Vision Transformers (ViTs)**, offer an alternative approach to image classification that could improve our model's ability to focus on critical regions of an image. By capturing long-range dependencies and learning more robust features from complex images, attention-based models have shown significant potential in image recognition tasks. This could lead to enhanced performance in dermatology models where fine-grained feature extraction is crucial for accurate diagnosis.

These future directions aim to further enhance the robustness, fairness, and generalization of dermatology AI models, ultimately contributing to a more equitable AI system in healthcare.

---

## **📜 References & Additional Resources**

* ["What is Fairness?" by Berkeley Haas](https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf)
* [TensorFlow Documentation](https://www.tensorflow.org)
* [Kaggle Tutorials on Skin Classification](https://www.kaggle.com/code/smitisinghal/skin-disease-classification)
* [Fine-tuning and Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
* [An Integrated Deep Learning Model with EfficientNet and ResNet for Accurate Multi-Class Skin Disease Classification](https://www.mdpi.com/2075-4418/15/5/551)


---
