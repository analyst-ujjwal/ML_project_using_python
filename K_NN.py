from sklearn.neighbors import KNeighborsClassifier

X = [
    [180,7],
    [200,7.5],
    [250,8],
    [300,8.5],
    [350,9],
    [360,9.5]
]
y = [0,0,0,1,1,1]
model = KNeighborsClassifier(n_neighbors =3)
model.fit(X,y)
weight = float(input('Enter weight of fruit:'))
size = float(input('enter size of fruit:'))
value = model.predict([[weight,size]])[0]
if value == 0:
    print(f'Based on weight:{weight} and size:{size}, the fruit is apple')
else:
    print(f'Based on weight:{weight} and size:{size}, the fruit is orange')