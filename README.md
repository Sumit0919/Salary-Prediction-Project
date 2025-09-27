# Salary-Prediction-Project
"A machine learning project to predict employee income."
# Employee Salary Prediction Project

This project uses a machine learning model to predict whether an individual's income is more or less than $50,000 based on data from the adult census dataset.

The analysis involves cleaning the data, comparing five different classification models, and selecting the best-performing one (Gradient Boosting) to build an interactive web application.

## Files in this Repository,hey////

* **`employee salary prediction (1).ipynb`**: The main Jupyter Notebook containing all the data cleaning, analysis, and model comparison.
* **`adult 3.csv`**: The dataset used for training and testing the models.
* **`app.py`**: A Python script for an interactive web application built with Streamlit.
* **`best_model.pkl`**: The saved Gradient Boosting model, which was the most accurate with approximately 85.7% accuracy.

## How to Run the Project

1.  **Prerequisites**: You must have Anaconda installed on your computer.
2.  **Run the Notebook**: Open the `employee salary prediction (1).ipynb` file in Jupyter Notebook and run all the cells to see the analysis.
3.  **Run the Web App**: Open the Anaconda Prompt/Terminal, navigate to this project's folder, and run the command: `streamlit run app.py`.
