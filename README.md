# Fantasy Baseball Prediction Model: Project Overview
- Created a ML model to predict MLB players’ offensive fantasy value for the next season given their past statistics to help in future fantasy baseball drafts.
- Compared output of model to historical ADP rankings to see if the model gave me an edge over ADP.
- Downloaded 9 years of historical data (2010-2018) from FanGraphs (link in Data Sources section below)
- Manipulated the raw data to create a "Fantasy Value"
- Optimized Linear, Elastic Net, Random Forest, XGBoost, LightGBM, and SVM models using GridSearchCV (LightGBM used BayesianOptimization) to reach the best model

## Code and Resources Use
**Python Version:** 3.7\
**Pacakges:** pandas, numpy, sklearn, seaborn, matplotlib, bayesian-optimization, xgboost, lightgbm\
**LightGBM Article:** https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm/comments \
**Project and GitHub Help:** https://www.youtube.com/watch?v=agHKuUoMwvY (Ken Jee)

## Data Sources
Raw Stats - https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2020&month=0&season1=2020&ind=0 \
ADP Data - https://fantasydata.com/mlb/fantasy-baseball-adp-rankings/hitters?season=2018&position=3

Pulled the data below from the sources listed above (somewhat arbitrarily grouped into bulleted items for readability):
- FanGraphs:
  - Season, Name, Team, Age, playerid
  - G, PA
  - HR, R, RBI, SB, AVG
  - BB%, K%
  - ISO, BABIP, OBP, SLG
  - wOBA, wRC+, Off
  - EV
  - BsR
  - O-Swing%, Z-Swing%, O-Contact%, Z-Contact%
  - GB/FB, LD%, GB%, HR/F
