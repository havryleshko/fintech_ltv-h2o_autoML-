### FinTech Customer LTV Prediction using H2O AutoML

## Predict your customer LTV with the financial data of the company:
https://fintech-ltv-h2o-automl.onrender.com/gradio/

## *Goal:*
to predict the Customer Lifetime Value (LTV) for a FinTech company using automated machine learning with H2O AutoML. 

## *Target:*
df['LTV']

## *Issues:*
The model’s predictions for customer LTV are inaccurate and systematically underestimate high-value customers. The scatter plot of actual vs. predicted LTV shows a wide spread and significant underprediction at higher LTV values. This indicates that, after removing redundant and leaky features, the remaining features are only weakly predictive of LTV—a common challenge in real-world LTV modeling.

### *side note:* 
this is a dataset problem, not a modeling problem.

**The dataset can be found within the repository.**

## *Tools used:*
-> Python
-> H2O AutoML
-> pandas, seaborn, matplotlib

**In many real-world business problems, R² between 0.2 and 0.5 is common for LTV, unless you have very rich behavioral or transactional data. My R2 has resulted in 0.36, which is honest and realistic outcome.**

## Pipeline:
**1. Data Loading & Cleaning**
-> Dropped features causing data leakage (Total_Spent, Avg_Transaction_Value, etc.)
-> Handled nulls

**2.EDA & Feature Engineering**
-> Visualizations (histograms, scatter plots, heatmaps)
-> Created feature interactions (e.g., Active_Days * App_Usage_Frequency)

**3. Model Training (H2O AutoML)**
-> Split data: 80% train, 10% valid, 10% test
-> H2OAutoML with max_models=10, nfolds=5

**4. Model Evaluation**
-> Best model selected from leaderboard
-> Evaluated using: R2, RMSE, MAE, MSE

**5. Visual Comparison**
-> Compared metrics with vs without data leakage

**Install dependencies.** (*requirements.txt*)

## Lesson I have learned from this project:
1. Choose the dataset wisely, garbage in = garbage out. **AutoML skips assumptions. I have to guide it.**
2. Write case studies and learn how to deploy the model into the business settings; also, learn how to write **case studies** for better understanding of potential business value in simple language.
3. AutoML automates **modeling, not thinking**.
4. Logging as a better way to **document**.





