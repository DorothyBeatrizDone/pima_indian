import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# View a sample of the dataframe
print(df.head())

# Get summary statistics
print(df.describe())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Do we have a class imbalance? Yes! About 35% of people of diabetes!
print("Do we have a class imbalance?")
print(y.value_counts(normalize=True))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random state is the seed. Otherwise, each code run will run a different test.

scaler = StandardScaler() #Uses a z-score to normalize the data.

# For numerical values, we always want to normalize the data to ensure that all features have the same scale.
# Otherwise, one feature may have more "weight" than another feature.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# We are using a logistic regression model. This type of model using a sigmoid function to classify classes as being "true" or "false." That is,
# logisitic regresion is a classification problem under "supervised learning"

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# As we can see in the classification report, we have a high recall. Yay!
print("Recall:", recall_score(y_test, y_pred))

# A confusion matrix is a great way to assess "false positives" and "false negatives"
# We have a class imbalance. That is, most people don't have diabetes. Accordingly, we must 
# choose a metric other than "accuracy" to address this class imbalance. 
# When the number of "positive" cases (ie those with diabetes) is substantially smaller than 
# the number of "negative" cases, we have a class imbalance. This can be directly observed
# within the confusion matrix. When false negatives are costly, we should prioritize using 
# "recall" (TPR) as opposed to "precision." 

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# We have 18 false negatives (actual 1s predicted as 0s). Yikes! 


# We should also prioritize using a precision-recall 
# plot over using an ROC curve, where the latter is not useful when addressing these class imbalances.

y_probs = model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
avg_precision = average_precision_score(y_test, y_probs)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (AP = {avg_precision:.2f})")
plt.grid(True)
plt.show()

# Because the number of people with diabetes is WAY smaller than those WITHOUT diabetes, we need to address this class imbalance.
# Note that having false negatives are more costly than false positives. Imagine if we told someone with diabetes that they DID NOT have diabetes. 
# Then they would not receive the treatment they urgently need!! 
# 
# Accordingly, we need to change the default 50-50 threshold so that we minimize the number of false negatives. 
# That is, decreasing the classification threshold tends to increase the number of false positives and decrease the number of false negatives. 
#  So let's experiment with varying thresholds!

thresholds = np.linspace(0.1, 0.9, 50)
recalls = [recall_score(y_test, y_probs >= t) for t in thresholds]
precisions = [precision_score(y_test, y_probs >= t) for t in thresholds]

# We can see that precision and recall move in opposite directions.
# There is a tradoff between these evaluation metrics. 
# Deciding between these tradeoffs are often problem specific. Are we more tolerant towards FP or FN? 
# As mentioned, we FN are intolerable, so we prioritize recall.

plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, precisions, label="Precision")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall at Different Thresholds")
plt.legend()
plt.grid(True)
plt.show()


# Let's assess the relative importance of each feature in determining whether diabetes
# Odds Ratio > 1: Increases the odds of having diabetes
# Odds Ratio < 1: Decreases the odds of having diabetes

coeffs = pd.DataFrame({
    "Feature": columns[:-1],
    "Coefficient": model.coef_[0],
    "Odds Ratio": np.exp(model.coef_[0])
})
print(coeffs.sort_values("Odds Ratio", ascending=False))

# In logistic regression, we use 
# 1. L2 Regularization
# 2. Log loss. Compared to linear regression, we use log loss to account for the lack of linearity in the function sigmoid.

# We want to minimize complexity. Accordingly, we are going to use min (loss + phi * regularization), where phi is the regularization rate
# Note regularization and learning rate move in opposite directions. 