import h2o
from h2o.automl import H2OAutoML
import logging
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

h2o.init(max_mem_size='4G')

df = h2o.import_file('/Users/ohavryleshko/Documents/GitHub/AutoML/FinTechLTV/digital_wallet_ltv_dataset.csv')

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')
# inspecting loaded data
logging.info('Begin data inspection')
logging.info('First 5 rows of data')
print(df.head(5))
#logging.info('Describing the data...')
#print(df.describe())
logging.info('List of all features...')
print(df.columns)
#logging.info('Shape...')
#print(df.shape)

#EDA (pre)
logging.info('Begin EDA...')
# dropping unnecessary features
df = df.drop(['Customer_ID', 'Total_Spent','Avg_Transaction_Value', 'Total_Transactions']) # as I ran the program a couple times, these features appear to be nearly perfectly correlated with LTV (0.9999) or can reconstruct mathematically Total_Spent, so I assume it causes data leakage.

# correlation
logging.info('Finiding the most correlated features...')

# converting into numerical using h2o
df['Location'] = df['Location'].asfactor()
df['Income_Level'] = df['Income_Level'].asfactor()
df['App_Usage_Frequency'] = df['App_Usage_Frequency'].asfactor()

# TEMPORARILY converting h2o into pandas for EDA
df_pd = df.as_data_frame(use_pandas=True)

#checking for skewed features
skewed = df_pd.select_dtypes(include='number').drop(columns=['LTV'], errors='ignore').skew()
print(f'Sroted skewed features:\n', skewed.sort_values(ascending=False))
print(df_pd.select_dtypes(include='number').corr()['LTV'].sort_values(ascending=False))

#dropping nulls to be on a safe side
logging.info('Checking for nulls...')
print(df_pd.isnull().sum())
df_pd = df_pd.dropna()

#IMPORTANT: converting back to h2o dataframe
df = h2o.H2OFrame(df_pd)
df['Location'] = df['Location'].asfactor()
df['Income_Level'] = df['Income_Level'].asfactor() 
df['App_Usage_Frequency'] = df['App_Usage_Frequency'].asfactor()

#EDA Main
#visualisations
#histogram
logging.info('Building a histogram for shape of data distribution...')
sns.set_theme(style='whitegrid')
df_pd.hist(figsize=(10, 6), bins='sturges', color='skyblue', grid=True, edgecolor='black')
plt.suptitle('Histogram for all features')
plt.tight_layout()
plt.show()

#function for scatterplot
scatter_features = ['Active_Days', 'Last_Transaction_Days_Ago', 'Min_Transaction_Value', 'Max_Transaction_Value', 'Loyalty_Points_Earned', 'Customer_Satisfaction_Score']
sns.set_theme(style='whitegrid')

def scatter_plot(df_pd, feature, target):
    for f in feature:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=f, y=target, data=df_pd, color='teal', alpha=0.6)
        plt.title(f'Scatterplot of {f} by {target}')
        plt.xlabel(f)
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()
        

scatter_plot(df_pd, scatter_features, 'LTV')

#heatmap correlation
heat_f = ['Active_Days', 'Last_Transaction_Days_Ago', 'Min_Transaction_Value', 'Max_Transaction_Value', 'Loyalty_Points_Earned', 'Customer_Satisfaction_Score']
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.set_style('white')
corr_heatmap = df_pd[heat_f].corr()

sns.heatmap(corr_heatmap, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap - top 6 features', fontsize=15)
plt.xlabel('Top 6', fontsize=14)
plt.xticks(ha='right', fontsize=10)
plt.ylabel('LTV', fontsize=14)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
logging.info('FInished EDA main.')

logging.info('Begin feature engineering...')
#Feature Engineering
##i use interactions as in real world data it it nearly always common that features give powerful outcome in combination, not isolation 
df_pd['App_Usage_Frequency_n'] = df_pd['App_Usage_Frequency'].astype('category').cat.codes
interactions = [
    ('App_Usage_Frequency_n', 'Active_Days'), # how often do they use bank card and how much on average they spent
    ('Max_Transaction_Value', 'Min_Transaction_Value'), # difference can give insight
    ('Customer_Satisfaction_Score', 'Issue_Resolution_Time'), # the faster, the better for lower LTV
    ('Active_Days', 'Customer_Satisfaction_Score'), # how much spent on average over the period of active days
    ('Last_Transaction_Days_Ago', 'Active_Days') # how much time has gone since last transaction during active days period
]

def add_interactions(df_pd, interactions):
    for f1, f2 in  interactions:
        new_col = f'{f1}_&_{f2}'
        df_pd[new_col] = df_pd[f1] * df_pd[f2]
    return df_pd # returning as separate feature

df_pd = add_interactions(df_pd, interactions)
df = h2o.H2OFrame(df_pd)
df['Location'] = df['Location'].asfactor()
df['Income_Level'] = df['Income_Level'].asfactor()

#creating new 
logging.info('Finished feature engineering.')

#start of modeling stage using h2o autoML (for the first time!)
logging.info('Begin modeling with h2o (autoML)...')

#splitting the data:
train, valid, test = df.split_frame(ratios=[0.8, 0.1], seed=42)

target = 'LTV'
features = df.columns
features.remove(target)

# to initialize autoML:
aml = H2OAutoML(
    max_runtime_secs=300, #maximum runtime to run models through
    max_models=10, #maximum numbers of models
    seed=42, #the same as random_state in scikit-learn; to get same results every time i run the program
    nfolds=5, #number of cross-validations
    keep_cross_validation_predictions=True #save CV predictions for stacking later
)

#to train the autoML model:
aml.train(x=features, y=target, training_frame=train, validation_frame=valid) # training_frame is is training set to train, validatino_frame -> 0.1 (in my case) for validation (hyperparameter tuning, model selection)

# to view leaderboard (summary table of model ranks)
logging.info('Creating leaderboard and printing out top 5 models...')
lbd = aml.leaderboard # this creates a leaderboard of all the models created (line 128 - max_models)
print('Top 5 models from leaderboard:', lbd.head(5)) # this gives a list of top 5 best performing ones

logging.info('Finding the best model...')
best_model = aml.leader
print('Best model is:', best_model)

perf = best_model.model_performance(test_data=test)
logging.info('Performance metrics loading...')
print('MSE:', perf.mse())
print('RMSE:', perf.rmse())
print('R^2:', perf.r2())
print('MAE:', perf.mae())

# checking model prediction
preds = best_model.predict(test)
preds_pd = preds.as_data_frame()
actuals_pd = test[target].as_data_frame()

# visualising 
#data for barplot (results )
experiments = ['With leakage', 'Without leakege']
r2_scores = [0.99, 0.36]
rmse_scores = [1000, 35000]
mae_scores = [800, 25000]

#barplot function
def bar_plot(labels, value, metric_name):
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 8))
    sns.barplot(x=labels, y=value, palette='Blues_d', edgecolor='black', width=0.5)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f'{metric_name} comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

bar_plot(experiments, r2_scores, 'R2 score comparison')
bar_plot(experiments, rmse_scores, 'RMSE score comparison')
bar_plot(experiments, mae_scores, 'MAE score comparison')