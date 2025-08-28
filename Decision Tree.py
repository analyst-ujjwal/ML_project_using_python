from sklearn.tree import DecisionTreeClassifier
X = [
    [7,2],
    [8,3],
    [9,8],
    [10,9]
]

y = [0,0,1,1]
model = DecisionTreeClassifier()
model.fit(X,y)
size = float(input('Enter size of fruit:'))
shade = float(input('enter color shade 1-10;'))
value = model.predict([[size,shade]])[0]
if value == 0:
    print(  f'Based on size:{size} and shade:{shade}, the fruit is apple')
else:
    print(f'Based on size:{size} and shade:{shade}, the fruit is orange')