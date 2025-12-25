# UFC-Fight-Outcome-Predictor
***
A project where historical UFC fight data is used to train predictive models for fight outcomes and then compare and evaluate the different models trained.

The project demonstrates data preprocessing, feature engineering, model training and evaluation, and interpretability using SHAP values.

***
## Dataset
The dataset includes all UFC fights from 1994 - September 2025. The model itself was only trained on fights post November 17, 2000 as this is when the unified rules for MMA were adopted by the UFC.

Note: The dataset was mirrored to remove corner bias, so each fight appears twice (once with each fighter as "red").

Source for data: https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025

***
## Feature Engineering
Key feature engineering steps:
- Cumulative stats: Total and average values for strikes, takedowns, submissions, and control time.
- Advantage features: Differences between the two fighters’ stats (e.g., r_sig_str_landed - b_sig_str_def).
- Rolling averages: For the last 3 fights of each fighter, calculated offensive and defensive averages for significant stats.
- Missing value handling: Flags for missing heights, reach, or date of birth; numerical missing values filled with 0.
- Derived metrics: Accuracy (landed / attempted) and defense ratios calculated on the fly.

***
## Model
Algorithm: LightGBM classifier
Hyperparameter tuning: Randomized search across multiple hyperparameters including num_leaves, learning_rate, subsample, colsample_bytree, and reg_lambda.

Evaluation metric: AUC (Area Under the Receiver Operating Characteristic Curve)

Final score: Test AUC = 0.654, a modest improvement from the baseline

***
## Model Interpretation
SHAP analysis revealed:
- Fighter age and reach differences are strong predictors of outcome.
- Offensive metrics such as significant strikes landed to the head and takedown attempts also contribute meaningfully.
- Missing value flags (e.g., reach_missing) initially introduced data leakage and were removed.
- Rolling averages of the last 3 fights offered temporal insight but did not significantly improve AUC.

*** 
## Lessons Learned
- Time-aware splits: In order to manage data leakage, I had to incrementally build out fighter stats and split the training, validation, and test data by time in order to prevent the model from being trained on future outcomes.
- Dataset versioning: I noticed at times adding new features I thought would improve the model's performance did not, as such it may have been beneficial to store different variations of the dataset to be used to train the models (or alternatively, create a list of core features for training and drop the rest). 
- Reducing Corner Bias: Since fighters were classified into red or blue corners, this introduced corner bias (e.g. the champion of a division is always in the red corner). In order to reduce this, I mirrored the fights but swapped the red and blue corners for a more robust prediction. 

***
## Future Improvements
- Explore additional rolling/trend features over more fights.
- Test other model architectures (XGBoost, neural networks).
- Investigate feature selection to reduce dataset size and training time.
