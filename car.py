import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

file_path1='E:\\Chai\\used_car_dataset.csv'
data=pd.read_csv(file_path1)
data

def convert_price(price_str):
    price_str = price_str.replace('₹', '').replace(',', '').replace(' ', '')
    if 'Lakh' in price_str:
        return float(price_str.replace('Lakh', '')) * 100000
    elif 'Crore' in price_str:
        return float(price_str.replace('Crore', '')) * 10000000
    return float(price_str)

data = data.drop_duplicates()
data

data['car_price_in_rupees'] = data['car_price_in_rupees'].apply(convert_price)


data['kms_driven'] = data['kms_driven'].str.replace('km', '').str.replace(',', '').astype(float)


data = pd.get_dummies(data, columns=['fuel_type', 'city'], drop_first=True)


data['car_age'] = 2023 - data['year_of_manufacture']
data = data.drop(columns=['year_of_manufacture', 'car_name'])
data

numerical_cols = ['kms_driven', 'car_age']
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])


X = data.drop('car_price_in_rupees', axis=1)
y = data['car_price_in_rupees']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test

lin_reg = LinearRegression()


lin_reg.fit(X_train, y_train)


y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.4f}")


plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()
