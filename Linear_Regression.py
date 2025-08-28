from sklearn.linear_model import LinearRegression
X = [[1],[2],[3],[4],[5]]
y = [40,50,65,75,90]
model = LinearRegression()
model.fit(X, y)
hours = float(input('Enter study hours: '))

prdicted_marks = model.predict([[hours]])
print(f'Based on your study hour:{hours}, you may score around : {prdicted_marks}')