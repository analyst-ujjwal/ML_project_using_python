import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/Users/ujjwalkumar/Desktop/Student.csv')
X = data[['StudyTimeWeekly']]
y = data['GradeClass']

model = LinearRegression()
model.fit(X, y )
predicted_grade = model.predict(X)

mae = mean_absolute_error(y, predicted_grade)
mse = mean_squared_error(y, predicted_grade)
rmse = np.sqrt(mse)
r2 = r2_score(y, predicted_grade)
print(f'Mean Absolute Error (MAE): {round(mae,2)}')
print(f'Mean Squared Error (MSE): {round(mse,2)}')
print(f'Root Mean Squared Error (RMSE): {round(rmse,2)}')
print(f'R-squared (MODEL ACCURACY): {round(r2,2)}')


plt.figure(figsize=(10,6))
plt.hist(data['GradeClass'], bins = 30 , color = 'skyblue', edgecolor = 'black')
plt.title('Distribution of Student Grades')
plt.xlabel('GradeClass')
plt.ylabel('Number of students')
plt.grid(True)
plt.show()


plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label= 'Actual Grades')
plt.plot(X, predicted_grade, color='red', label = ' predicted Grades (Regression Line)')
plt.title('Modeling Student Grades Based on Study Time Weekly')
plt.xlabel('Study Time Weekly (hours)')
plt.ylabel('Final GradeClass')
plt.grid(True)
plt.show()

new_grade = 5
predicted_new_grade = model.predict([[new_grade]])
print(f'Final grade for prdeicted {new_grade} is {predicted_new_grade} Grades')
