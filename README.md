# FinalProject
	1.	Dataset Exploration & Preprocessing
	•	The dataset contained structured tabular data with clinical, cognitive, and lifestyle attributes.
	•	No missing values were found, simplifying preprocessing.
	•	Categorical features were encoded:
	 •	Nominal variables  → One-Hot Encoding.
	 •	Ordinal variables  → Label Encoding.
	2.	Model Selection & Training
	•	Classification models tested:
	 •	Decision Tree
	 •	K-Nearest Neighbors (KNN)
	 •	Support Vector Machine (SVM)
	 •	Random Forest
	 •	Extreme Gradient Boosting (XGBoost) – Best Performer (95% Accuracy)
	 •	Feature selection improved performance (e.g., SelectKBest, SelectFromModel).
	3.	Hyperparameter Tuning
	•	RandomizedSearchCV and GridSearchCV were used to optimize:
	 •	n_estimators, max_depth, learning_rate, subsample, etc.
	•	Class balancing techniques (scale_pos_weight) were tested but had no significant impact due to a moderate class imbalance (65% healthy, 35% AD).
	4.  Feature Importance Analysis
	•	Extracted from XGBoost’s best model.
	•	Features related to cognitive and functional assessments (MMSE, ADL, Functional Score, etc.) had the highest impact.
	•	Observed that binary features (e.g., Memory Complaints, Behavioral Problems) might appear less frequently in decision trees but still play a crucial role.
        5. Loading & Deploying the Model (Joblib)
        6. Web App Creation Using Streamlit
