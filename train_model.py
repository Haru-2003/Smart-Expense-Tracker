import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
dataset_path = "../dataset/Personal_Finance_Dataset.csv"
data = pd.read_csv(dataset_path)

# Drop null values
data.dropna(inplace=True)

# Extract features and labels
X = data["Description"]
y = data["Category"]

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model and vectorizer
model_path = "../models/expense_model.pkl"
vectorizer_path = "../models/vectorizer.pkl"

pickle.dump(model, open(model_path, "wb"))
pickle.dump(vectorizer, open(vectorizer_path, "wb"))

print(f"Model saved to {model_path}")
print(f"Vectorizer saved to {vectorizer_path}")
