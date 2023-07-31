## Adult Income Classification Using Machine Learning Classifiers - Jupyter Notebook

This Jupyter Notebook is designed to perform a classification task on the Adult Income dataset. The goal is to predict whether an individual earns more than $50K per year or not based on various features such as age, education, occupation, and more. The notebook uses several machine learning classifiers to achieve this task.

### Prerequisites

Before running the notebook, ensure you have the following installed:

- Jupyter Notebook
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Dataset

The notebook assumes that you have the "adult.csv" file in the same directory as the notebook. The dataset contains information about individuals, with each row representing a person and columns for various features.

### Functionality

The notebook is divided into several code cells, each with specific functionality:

1. **Data Loading and Preprocessing**: This section loads the dataset, replaces column names with custom names, and creates dummy variables for categorical features using one-hot encoding.

2. **Data Splitting**: The dataset is divided into training and testing sets using a 75:25 ratio.

3. **Decision Tree Classifier**: This section trains a Decision Tree classifier using the training data and evaluates its performance on the testing data.

4. **Random Forest Classifier**: This section trains a Random Forest classifier using the training data and evaluates its performance on the testing data.

5. **Logistic Regression**: This section trains a Logistic Regression classifier using the training data and evaluates its performance on the testing data.

6. **K-Nearest Neighbors (KNN) Classifier**: This section trains a KNN classifier using the training data and evaluates its performance on the testing data.

7. **Support Vector Machine (SVM) Classifier**: This section trains an SVM classifier with a linear kernel using the training data and evaluates its performance on the testing data.

8. **Comparison and Visualization**: The notebook compares the accuracy of all classifiers and displays a bar graph to visualize their performance.

### Running the Notebook

To run the notebook, follow these steps:

1. Ensure you have all the prerequisites installed on your system.

2. Download the "adult.csv" file and place it in the same directory as the notebook.

3. Launch Jupyter Notebook on your system.

4. Navigate to the directory containing the notebook.

5. Open the "Adult_Income_Classification.ipynb" notebook.

6. Execute each code cell sequentially by clicking on the cell and pressing "Shift + Enter" or use the "Run" button in the Jupyter toolbar.

7. The script will print the confusion matrix, classification report, and accuracy for each classifier. Additionally, a bar graph showing the accuracy scores of each classifier will be displayed.

### Conclusion

Based on the performance evaluation, the notebook identifies the Logistic Regression classifier as having the best accuracy for this particular classification task. The bar graph provides a visual comparison of the accuracy scores achieved by different classifiers.

