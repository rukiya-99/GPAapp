import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

# Load the data from CSV
data = pd.read_csv('student_performance_data.csv', encoding='ISO-8859-1')

# Ensure the data is in DataFrame format
df = pd.DataFrame(data)

# Features and target
X = df.drop(columns=['StudentID', 'GPA'])
y = df['GPA']

# Preprocessing categorical data and numerical data
categorical_features = ['Gender', 'Major', 'PartTimeJob', 'ExtraCurricularActivities']
numerical_features = ['Age', 'StudyHoursPerWeek', 'AttendanceRate']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Add a constant (intercept) to the processed features for the OLS model
X_train_processed_with_constant = sm.add_constant(X_train_processed)

# Fit the OLS model for p-value calculation and to get the model summary
ols_model = sm.OLS(y_train, X_train_processed_with_constant).fit()

# Output the model summary, including p-values
print("\nOLS Model Summary:")
print(ols_model.summary())

# Print OLS model parameters
print("\nOLS Model Parameters:")
print(ols_model.params)

# Train the scikit-learn model pipeline with the preprocessed data
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the pipeline model
model_pipeline.fit(X_train, y_train)

# Predicting
predictions = model_pipeline.predict(X_test)

# Model performance
score = model_pipeline.score(X_test, y_test)
print(f"\nModel R^2 score: {score:.2f}")

# Summary of predictions vs actual values
summary_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print("\nSummary of Actual vs Predicted GPA:")
print(summary_df)

# Print scikit-learn model parameters
print("\nScikit-learn Model Parameters:")
print(model_pipeline.named_steps['regressor'].coef_)
print(model_pipeline.named_steps['regressor'].intercept_)

# Save the trained model to a file
joblib.dump(model_pipeline, 'student_gpa_model.pkl')
print("\nModel has been saved to 'student_gpa_model.pkl'")
