import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv('student_data.csv')
X = df[['Hours_Studied', 'Sports_Hours', 'Attendance']]
y = df['Pass']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl!")
