import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split

data ={
    'study_hour' : [1,2,3,4,5],
    'Test_score' : [40,50,60,70,80] 
}

df = pd.DataFrame(data)
 

 #Standard Scaler

Scaler = StandardScaler()
Scaled = Scaler.fit_transform(df)

print('Standard Scaler Output:')
print(pd.DataFrame(Scaled, columns= ['study_hour', 'Test_score']))

#MinMax Scaler
MMS = MinMaxScaler()
MMS_scaled = MMS.fit_transform(df)
print('\n MinMax Scaler Output:')
print(pd.DataFrame(MMS_scaled, columns = ['study_hour', 'Test_score']))

X=df[['study_hour']]
y= df[['Test_score']]

X_train, X_test , y_train, y_test = train_test_split(X,y , test_size =0.2, random_state = 42)
print('Training Data:')
print(X_train)

print('Test Data:')
print(X_test)

print('Training Data:')
print(y_train)

print('Test Data:')
print(y_test)

