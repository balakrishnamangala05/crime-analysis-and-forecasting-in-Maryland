import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
data_path = 'Transformed_Final_Data.csv'
data = pd.read_csv(data_path)

# Define the features and targets
X = data[['County', 'Year', 'Grades Pre-K', 'Grades K-5', 'Grades 6-8', 'Grades 9-12', 'Unemploy_Value', 'POPULATION', 'B & E', 'LARCENY THEFT', 'M/V THEFT', 'GRAND TOTAL', 'PROPERTY CRIME TOTALS']]
y = data[['MURDER', 'RAPE', 'ROBBERY', 'AGG. ASSAULT', 'VIOLENT CRIME TOTAL']]

# Create a Column Transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Year', 'Grades Pre-K', 'Grades K-5', 'Grades 6-8', 'Grades 9-12', 'Unemploy_Value', 'POPULATION', 'B & E', 'LARCENY THEFT', 'M/V THEFT', 'GRAND TOTAL', 'PROPERTY CRIME TOTALS']),
        ('cat', OneHotEncoder(sparse_output=False), ['County'])
    ])

# Create a preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_processed = pipeline.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define the base models
estimators = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# Initialize Stacking Regressor with a Linear Regression meta-regressor
stacking_regressor = StackingRegressor(
    estimators=estimators, 
    final_estimator=LinearRegression(),
    passthrough=True
)

# Wrap the stacking regressor with MultiOutputRegressor
multioutput_regressor = MultiOutputRegressor(stacking_regressor)

# Fit the model
multioutput_regressor.fit(X_train, y_train)

# Predict on test set and evaluate
predictions = multioutput_regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions, multioutput='raw_values')
mae = mean_absolute_error(y_test, predictions, multioutput='raw_values')

# Print results for each selected crime type
crime_types = ['MURDER', 'RAPE', 'ROBBERY', 'AGG. ASSAULT', 'VIOLENT CRIME TOTAL']
for i, crime in enumerate(crime_types):
    print(f"Stacking Model for {crime} - MSE: {mse[i]}, MAE: {mae[i]}")

# Calculate average MSE and MAE
average_mse = np.mean(mse)
average_mae = np.mean(mae)
print(f"Average Stacking Model - MSE: {average_mse}, MAE: {average_mae}")

# Save the model and the preprocessing pipeline for later use
joblib.dump(multioutput_regressor, 'multioutput_stacking_regressor.pkl')
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')
