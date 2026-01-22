# Iris-Flower-Classification-Supervised-ML

#  Iris Flower Classification using Supervised Machine Learning

 **Author:** Ishwor Chandra Paudyal (ICP)
 **Email:** ishwor.paudyal12@gmail.com  
  
  

---

##  Project Overview

A botanical research centre aims to automate the identification of Iris flower species based on physical measurements such as sepal and petal dimensions. Manual identification is:

- Time-consuming  
- Error-prone  
- Not scalable  

This project builds a machine learning classification system to automatically predict the species of Iris flowers using supervised learning algorithms.

---

##  Project Objectives

- Implement and evaluate three classification models:
  - K-Nearest Neighbours (KNN)
  - Logistic Regression
  - Naive Bayes
- Compare model performances using accuracy, precision, recall, and F1-score
- Identify the best-performing model
- Visualize results using confusion matrices

---

##  Project Structure

Iris-Flower-Classification-Supervised-ML/
│
├── README.md
├── data/
│ └── Iris.csv
│
├── notebooks/
│ └── iris_classification.ipynb
│
├── reports/
│ └── Report.pdf
│
├── results/
│ ├── confusion_matrix_knn.png
│ ├── confusion_matrix_logistic.png
│ └── confusion_matrix_naive_bayes.png
│
├── requirements.txt


---

##  Dataset Description

The dataset used is the classic Iris dataset containing 150 samples equally distributed among three species:

| Feature | Description |
|--------|-------------|
| SepalLengthCm | Sepal length (cm) |
| SepalWidthCm | Sepal width (cm) |
| PetalLengthCm | Petal length (cm) |
| PetalWidthCm | Petal width (cm) |
| Species | Target class (Setosa, Versicolor, Virginica) |

---

##  Tools & Technologies

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

##  Machine Learning Models Used

###  K-Nearest Neighbours (KNN)
- Distance-based classification
- No assumptions about data distribution
- Performed best in this project

###  Logistic Regression
- Linear probabilistic classifier
- Used One-vs-Rest strategy for multi-class classification

###  Naive Bayes
- Probabilistic classifier based on Bayes’ Theorem
- Gaussian Naive Bayes used for continuous data

---

##  Data Preprocessing Steps

- Removed unnecessary columns
- Encoded categorical target labels
- Split dataset:
  - 50% training data
  - 100% testing data (as per assignment instruction)
- Standardized features where required

---

##  Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

##  Best Model Selection

**Best Model:** K-Nearest Neighbours (KNN)

### Reasons:
- Highest accuracy
- Simple and effective for small datasets
- No distribution assumptions

---

##  Conclusion

This project demonstrates the application of supervised machine learning algorithms for real-world classification problems. All three models achieved high accuracy due to the clean and balanced nature of the dataset. KNN achieved the best performance and was selected as the final model.

---

##  References

- Scikit-learn Documentation: https://scikit-learn.org/
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.*

---

⭐ If you like this project, feel free to star the repository!

