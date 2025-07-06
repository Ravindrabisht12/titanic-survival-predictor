
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('data/train.csv')
train_df.head()


sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.show()


# Fill missing Age values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# Fill missing values
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Convert categorical columns into dummy variables
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop unnecessary
train_df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])



# Prepare the data for training
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))


