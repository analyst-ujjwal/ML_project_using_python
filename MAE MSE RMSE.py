from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

real_score = [90,60,80,100]

predicted_score = [85,70,70,95]
mae = mean_absolute_error(real_score, predicted_score)
mse = mean_squared_error(real_score, predicted_score)
rmse = np.sqrt(mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)