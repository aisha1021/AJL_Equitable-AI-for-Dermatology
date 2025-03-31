# BTT-Spring-AI-Studio-25
# GitHub AJL Kaggle Competition!
---

### **👥 Team Members**

Aisha, Ana, Mysara, Rishita, Yousra, Zohreh


---

## **🎯 Project Highlights**

* Built an ensemble model using transfer learning and finetuning of existing models to perform image classification on pictures of skin disease
* Achieved a ranking of 2th place virtual team, 5th place on Kaggle leaderboard
* Implemented data augmentation and image preprocessing techniques to optimize results of imbalanced data


🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

1. Cloning the Repository
To clone the repository to your local machine, use the following Git command:
git clone https://github.com/yawad2/BTT-Spring-AI-Studio-25.git 


2. Accessing The Datasets
The dataset for this competition can be located here: https://www.kaggle.com/competitions/bttai-ajl-2025/data. Navigate to the Download button and then extract it into a folder in your repository directory.

3. Running the Notebook

To run the Jupyter Notebook:
Make sure the environment is set up and dependencies are installed.
Launch Jupyter Notebook:
jupyter notebook
Alternatively, open the notebook file (your_notebook.ipynb) in your browser, and run the cells sequentially.
---

## **🏗️ Project Overview**

This competition is part of the Break Through Tech AI program, in collaboration with the Algorithmic Justice League. It focuses on addressing a huge issue in modern day dermatology software tools: their underperformance for people with darker skin tones. In participating in this challenge, we aimed to leverage AI to help build a more accurate and equitable dermatology AI tool. 

The main objective of this competition was to train a machine learning model capable of running image classification to classify 21 different skin conditions across various skin tones. 

---

## **📊 Data Exploration**

**Describe:**

* The dataset used is the data provided in Kaggle by AJL. It’s a subset of the FitzPatrick17 dataset, which is a collection of around 17,000 images depicting a variety of serious and cosmetic dermatologic conditions. 
Used the official Kaggle dataset containing labeled dermatology images and metadata (including Fitzpatrick skin type)
Explored class imbalance across both skin condition labels and skin tone categories
Visualized:
Label distribution
Skin tone distribution
Sample images per class
Correlation heatmaps and histogram



---

## **🧠 Model Development**


* Model(s) used 

We built and trained a variety of different models, including ResNet50, MobileNet, EfficientNet, and DenseNet. Our method with training all these models was building each slowly and saving them after a certain run of epochs, then downloading them to save the history of their training. We would load them in when we wanted to train them on more epochs to be efficient.  After doing this with all the models, we ensemble the models together as a means of comparing their use using both hard and soft voting methods. Soft voting combines and averages the predictions of the models and chooses the model with the highest average prediction. Hard voting simply chooses the model with the highest votes in the prediction. 

Ensembling ResNet50, EfficientNetB3, EffiecientNetB4, and DenseNet121 using hard voting yielded the best result and highest accuracy of our attempts. 

* Training setup 
Our training setup was 80% training and 20% validation. We used F1-score as our evaluation metric. For our baseline performance we measured using deep learning models pre-trained on large image datasets.

---

## **📈 Results & Key Findings**


* Performance metrics 
Kaggle Leaderboard score: We oscillated between 3rd and 4th place on our final submissions.


---

## **🖼️ Impact Narrative**

**AJL challenge:**

We tried to use data augmentation to balance the training samples of different skin tones. We also looked at model fairness using separate validation across the different skin tones. It could help improve dermatology for underrepresented skin tones which can translate into ethical issues in healthcare.

---

## **🚀 Next Steps & Future Improvements**


* What would you do differently with more time/resources?

We would potentially try using a CNN model that had not previously been trained on the ImageNet dataset. In doing so, we would be able to rescale the AJL dataset to be smaller sizes, and manipulate the CNN layers as necessary, as well as run for more epochs. This would potentially give us more flexibility and more efficiency in runtime. 

* What additional datasets or techniques would you explore?

Collect or incorporate external datasets with better representation of darker skin tones
Experiment with GAN-based image augmentation to synthetically balance rare classes

---

## **📄 References & Additional Resources**

"What is Fairness?" by Berkeley Haas: https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf
TensorFlow documentation: https://www.tensorflow.org
Kaggle tutorials on skin classification: https://www.kaggle.com/code/smitisinghal/skin-disease-classification

---

