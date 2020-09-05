# Fantasy Baseball Prediction Model: Project Overview
- Created a machine learning model to predict MLB players’ offensive fantasy value for the next season given their past statistics to help in future fantasy baseball drafts.
- Compared output of model to historical ADP rankings to see if the model gave me an edge over ADP.
- Downloaded 9 years of historical data (2010-2018) from FanGraphs (link in Data Sources section below).
- Manipulated the raw data to create a "Fantasy Value" which was used as the target value.
- Optimized Linear, Elastic Net, Random Forest, XGBoost, LightGBM, and SVM models using GridSearchCV (LightGBM used BayesianOptimization) to reach the best model.

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
![alt text](https://github.com/nkrajew/baseball_proj/blob/master/pictures/corr_matrix_image.png "Correlation Matrix")
![alt text](https://github.com/nkrajew/baseball_proj/blob/master/pictures/pair_plot_resized.png "Pair Plot")
![alt text](https://github.com/nkrajew/baseball_proj/blob/master/pictures/target_distribution.png "Target Distribution")

## Model Building
Based on the findings from the EDA, I created a data frame with only the most correlated features greatly reducing the total features to only six.

Next, I split the data into a training and test set based on Season. I set seasons 2010-2015 as the training set and seasons 2016 and 2017 as the test set. The number of useable seasons is one less than the number of seasons I pulled since the target value is for the "next year". 

*Note: Because I used an Elastic Net model (discussed next) which can correct for multicollinearity, a separate dataset retaining more features was created to build that model. Instead of being normalized, the features in this dataset were standardized in accordance with the Elastic Net requirements.*

I tried out six different models and evaluated them on Mean Squared Error (MSE). I chose MSE because I wanted the model to be sensitive to outliers since I believe missing out on a really great player could make or break a season. *Side note: MSE is also quicker for learning, however, my dataset is fairly small so I'm not sure there would have been a noticeable difference.*

I tried six different models to predict MLB offensive fantasy value (FV):
- **Multiple Linear Regression** &mdash; Baseline for the model.
- **Random Forest** &mdash; Mainly used because I was exploring different models.
- **XGBoost** &mdash; Same reason as Random Forest.
- **Elastic Net** &mdash; I wanted a model that could perform feature selection and correct for multicollinearity. Elastic Net is basically a best-of-both-worlds since it combines the effects of LASSO and Ridge Regression.
- **LightGBM** &mdash; Same reason as other decision tree models.
- **SVM** &mdash; I wanted experience with an SVM and since my data set was small I used it here.

## Model Performance
The Elastic Net model outperformed the other models.
- Linear Regression: MSE = 8.64
- Random Forest: MSE = 8.65
- XGBoost: MSE = 8.74
- **Elastic Net: MSE = 8.06**
- LightGBM: MSE = 9.95
- SVM: MSE = 8.71

## Comparison to ADP Rankings
I compared the results of the Elastic Net model against the pre-season rankings as determined by ADP. The two ranking systems were compared by placing each player into a group _x_ out of *n* groups based on their Predicted FV or ADP, whichever is applicable. The same will be done for Target FV. The prediction/rankings were then compared and scored. I only evaluated the top 50% of players (determined by Target FV) as they can be considered the league's "top performers" and thus would be players you would care about predicting correctly. 

The main evaluation is on a metric I called Top Performer Ratio which is:
*placeholder for image*
![alt text](https://github.com/nkrajew/baseball_proj/blob/master/pictures/TPR_formula.PNG "TPR Formula")

Other evaluation metrics used:
- **Missed Predictions**: The number of predictions that were incorrectly placed inside/outside of the top 50%
- **Missed Target Value**: The total Target Value of the players that were incorrectly predicted outside of the top 50% 
- **Value Found**: The total Target Value of players that were correctly predicted inside the top 50%
- **Total Value**: Value Found - Missed Target Value

### Results

![alt text](https://github.com/nkrajew/baseball_proj/blob/master/pictures/Results_vs_ADP.PNG "Results v ADP")

Disappointingly, the model performs worse than ADP rankings. However, this provides me with an opportunity to further tinker with the model and see if I can improve its predictive power!

## Potential Next Steps
- **Find additional features to add**
    - Career Stats
    - Injury Proneness (# of injuries)
    - Home Park (Team could be used as proxy)
    - League (AL vs. NL)
- **Investigate model's missed predictions**
    - Could help understand where performance is suffering compared to ADP
- **Change to classification model**
    - Since end comparison relies on properly categorizing players, a classification model might be better suited
    - Regression model could be useful when determining "how much better" a player is from the next
- **Align training penalty to end goal**
    - Results only care about top 50% right/wrong, so model should be optimized for correctly predicting FV for top performers and care less about correctly aligning FV on poor performers. 
