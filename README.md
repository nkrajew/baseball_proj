# Fantasy Baseball Prediction Model: Project Overview
- Created a machine learning model to predict MLB players’ offensive fantasy value for the next season given their past statistics to help in future fantasy baseball drafts.
- Compared output of model to historical ADP rankings to see if the model gave me an edge over ADP.
- Downloaded 9 years of historical data (2010-2018) from FanGraphs (link in Data Sources section below)
- Manipulated the raw data to create a "Fantasy Value" and the input dataset for the models
- Optimized Linear, Elastic Net, Random Forest, XGBoost, LightGBM, and SVM models using GridSearchCV (LightGBM used BayesianOptimization) to reach the best model

## Code and Resources Used
**Python Version:** 3.7\
**Packages:** pandas, numpy, sklearn, seaborn, matplotlib, bayesian-optimization, xgboost, lightgbm\
**LightGBM Article:** https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm/comments \
**Project and GitHub Help:** https://www.youtube.com/watch?v=agHKuUoMwvY (Ken Jee)

## Data Sources
Raw Stats - https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2020&month=0&season1=2020&ind=0 \
ADP Data - https://fantasydata.com/mlb/fantasy-baseball-adp-rankings/hitters?season=2018&position=3

Pulled the raw data below from the sources listed above (somewhat arbitrarily grouped into bulleted items for readability):
- FanGraphs:
  - Season, Name, Team, Age, playerid
  - PA (only pulled players with minimum 400 PAs)
  - G, HR, R, RBI, SB, AVG
  - BB%, K%
  - ISO, BABIP, OBP, SLG
  - wOBA, wRC+, Off
  - EV
  - BsR
  - O-Swing%, Z-Swing%, O-Contact%, Z-Contact%
  - GB/FB, LD%, GB%, HR/F
- FantasyData:
  - RK
  - ID
  - Name
  - ADP

## Data Cleaning and Creation (in Data Prep notebook)
After pulling the raw data, I needed to create some new variables and create the target variable. I needed to clean some of the data as well.
- Engineered the following data for evaluation:
  - HR_rate: HR/G
  - R_rate: R/G
  - RBI_rate: RBI/G
  - SB_rate: SB/G
  - HR_rate_std: standardized HR_rate by year
  - R_rate_std: standardized R_rate by year
  - RBI_rate_std: standardized RBI_rate by year
  - SB_rate_std: standardized SB_rate by year
  - AVG_std: standardized AVG by year
  - f_val_std: summation of above 5 standardized metrics
  - ny_f_val_std: the f_val_std for the player's next season (this is the target)
  - New Rank (used in ADP to re-rank after removing players that did not meet 400 PA minimum)
- Filtered out entries where ny_f_val_std was blank (due to players retiring, getting injured, etc.)

## EDA 
I investigated the correlation strength with the target, any potential collinearity, and the shape of the target distribution. I used these findings to adjust the data frames I used in the predictions notebook. Attached are some images of the analysis.\
![alt text](https://github.com/nkrajew/baseball_proj/blob/master/corr_matrix_image.png "Correlation Matrix")
![alt text](https://github.com/nkrajew/baseball_proj/blob/master/pair_plot.png "Pair Plot")
![alt text](https://github.com/nkrajew/baseball_proj/blob/master/target_distribution.png "Target Distribution")

## Model Building
Based on the findings from the EDA, I created a data frame with only the most correlated features greatly reducing the total features to only six.

Next, I split the data into a training and test set based on Season. I set seasons 2010-2015 as the training set and seasons 2016 and 2017 as the test set. The number of useable seasons is one less than the number of seasons I pulled since the target value is for the "next year". 

*Note: Because I use an Elastic Net model (discussed next) and it can correct for multicollinearity, a separate dataset retianing more features was used to build that model. Instead of being normalized, the fetures in this data set were standardized in accordance with the Elastic Net requirements.*

I tried out six different models and evaluated them on Mean Squared Error (MSE). I chose MSE because I wanted the model to be sensitive to outliers since I believe missing out on a really great player could make or break a season. *Side note: MSE is also quicker for learning, however, my dataset is fairly small so I'm not sure there would have been a noticeable difference.*

I tried six different models to predict MLB offensive fantasy value (FV):
- **Multiple Linear Regression** &mdash; Baseline for the model.
- **Random Forest** &mdash; Mainly used because I was exploring different models.
- **XGBoost** &mdash; Same reason as Random Forest.
- **Elastic Net** &mdash; I wanted a model that could perform feature selection. I also wanted to experiment and get practice with the Elastic Net model.
- **LightGBM** &mdash; Same reason as other decision tree models.
- **SVM** &mdash; I wanted experience with an SVM and since my data set was small I used it here.

## Model Performance
The Elastic Net model outperformed the other models.
- Linear Regression: MSE = 8.63
- Random Forest: MSE = 8.65
- XGBoost: MSE = 8.85
- Elastic Net: MSE = 8.03
- LightGBM: MSE = 9.75
- SVM: MSE = 8.71
