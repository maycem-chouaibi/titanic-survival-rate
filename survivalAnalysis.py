import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

def analyze():   
    # Load the cleaned data
    train_data = pd.read_csv('data/train_cleaned.csv')
    test_data = pd.read_csv('data/test_cleaned.csv')

    # Perform regression analysis
    X_train = train_data.drop(columns=['Survived'])
    y_train = train_data['Survived']
    X_test = test_data

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Print the model coefficients
    print(model.coef_)
    print(model.intercept_)

    # Predict the test data
    predictions = model.predict(X_test)

    # Print predictions
    passenger_ids = pd.read_csv('data/test.csv')['PassengerId']
    predictions = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})

    # Calculate accuracy
    y_true = pd.read_csv('data/gender_submission.csv')['Survived']
    y_pred = predictions['Survived']
    accuracy = (y_true == y_pred).mean()
    print(f'Accuracy: {accuracy}')

    # Save the predictions
    pd.DataFrame(predictions).to_csv('data/predictions.csv', index=False)

    return
