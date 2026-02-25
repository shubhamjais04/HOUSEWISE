HOUSEWISE - House Price Prediction
Hey! This is my machine learning project where I predict house prices using some basic property features. I built this to learn more about regression models and get hands-on with real data.
What This Project Does
Takes in details about a house (like bedrooms, bathrooms, square footage) and predicts how much it might sell for. I used data from King County, Seattle to train the model.
Why I Built This
I wanted to understand how machine learning actually works beyond just theory. House price prediction seemed like a good starting point because:

The problem is easy to understand
There's plenty of data available
I could actually see if my predictions made sense

What I Used:

Python - for everything
pandas & numpy - data cleaning and manipulation
matplotlib & seaborn - making visualizations
scikit-learn - building the models
Jupyter Notebook - running and testing everything

The Dataset
I used the King County House Sales dataset which has around 21,000 house sales. For this project, I focused on just 5 features to keep things simple:

Number of bedrooms
Number of bathrooms
Living area (sqft)
Lot size (sqft)
Number of floors

What I Did:

Cleaned the data - handled missing values, removed outliers
Explored patterns - made some plots to see how features relate to price
Tried different models - tested Linear Regression and Random Forest
Picked the best one - Random Forest worked better (R² = 0.87)
Saved the model - so I can use it later without retraining

How to Use It
If you want to run this:
bash# Clone the repo
git clone https://github.com/shubhamjais04/HOUSEWISE.git
cd HOUSEWISE

# Install what you need
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Open the notebook
jupyter notebook Housewise.ipynb
Then just run the cells in order. To make predictions with your own house data, scroll down to the prediction cells and change the values.
Results
The Random Forest model got about 87% accuracy on test data. Not perfect, but pretty decent for a basic model! The average prediction error is around $85,000.
Biggest learning: living area (sqft_living) matters way more than other features for predicting price.
What I Learned

How to clean messy real-world data
The difference between different regression models
Why Random Forest often beats Linear Regression
How to save and load models with pickle
Feature scaling is important!

Things I Want to Improve

Add more features like location, condition, year built
Try other algorithms like XGBoost
Maybe build a simple web app where people can input house details
Better visualization of predictions vs actual prices

Files in This Repo

Housewise.ipynb - main notebook with all the code
kc_house_data.csv - the dataset
models/ - saved model and scaler files
README.md - this file

Contact
If you have questions or suggestions, feel free to reach out!
Shubham Jaiswal
Email: shubhjais.in@gmail.com
LinkedIn: linkedin.com/in/shubhamjaiswal2004

Note: This was my first proper ML project, so the code might not be perfect. Always learning and improving!





