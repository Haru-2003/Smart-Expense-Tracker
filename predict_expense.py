import pickle

# Load the saved model and vectorizer
model = pickle.load(open("../models/expense_model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

# Function to predict expense category
def predict_expense(description):
    vectorized_text = vectorizer.transform([description])
    category = model.predict(vectorized_text)[0]
    return category

# Example prediction
if __name__ == "__main__":
    expense = "Paid monthly rent"
    predicted_category = predict_expense(expense)
    print(f"Expense: {expense}")
    print(f"Predicted Category: {predicted_category}")
