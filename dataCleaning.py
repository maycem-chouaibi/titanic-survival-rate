import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



def cleanData():
    # Load the data
    test_data = pd.read_csv('data/test.csv')
    train_data = pd.read_csv('data/train.csv')

    # Check for missing values
    print(train_data.isnull().sum())
    print(test_data.isnull().sum())

    # Replace String values with numerical values
    train_data['Sex'] = train_data['Sex'].replace('female', 1)
    train_data['Sex'] = train_data['Sex'].replace('male', 0)
    test_data['Sex'] = test_data['Sex'].replace('female', 1)
    test_data['Sex'] = test_data['Sex'].replace('male', 0)

    train_data['Embarked'] = train_data['Embarked'].replace('S', 0)
    train_data['Embarked'] = train_data['Embarked'].replace('C', 1)
    train_data['Embarked'] = train_data['Embarked'].replace('Q', 2)

    test_data['Embarked'] = test_data['Embarked'].replace('S', 0)
    test_data['Embarked'] = test_data['Embarked'].replace('C', 1)
    test_data['Embarked'] = test_data['Embarked'].replace('Q', 2)

    # # Drop NaN in Cabin
    # train_data = train_data.dropna(subset=['Cabin'])
    # test_data = test_data.dropna(subset=['Cabin'])

    # Replace NaN in Embarked with 1 if Survived is 0
    train_data.loc[(train_data['Embarked'].isnull()) & (train_data['Survived'] == 0), 'Embarked'] = 1

    # Drop rest of NaN values in Embarked
    train_data = train_data.dropna(subset=['Embarked'])

    # Replace NaN in Age with mean
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

    # Replace NaN in Fare with mean
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

    # Drop unused columns
    train_data = train_data.drop(columns=['Name', 'Ticket', 'Cabin'])
    test_data = test_data.drop(columns=['Name', 'Ticket', 'Cabin'])

    # # Print the first 5 rows of the dataframe
    # print(train_data.head())
    # print(test_data.head())

    # # Print data types
    # print(train_data.dtypes)
    # print(test_data.dtypes)

    # Save the cleaned data
    train_data.to_csv('data/train_cleaned.csv', index=False)
    test_data.to_csv('data/test_cleaned.csv', index=False)

    # # Print information about the data
    # print(train_data.info())
    # print(test_data.info())

    # Perform relative importance analysis
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    model = RandomForestClassifier()
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(X.shape[1]):
        # Print feature name and importance
        print(f"{f + 1}. {X.columns[indices[f]]}: {importances[indices[f]]}")

    

cleanData()