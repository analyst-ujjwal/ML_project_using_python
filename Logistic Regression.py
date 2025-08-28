from sklearn.linear_model import LogisticRegression
X = [[1],[2],[3],[4],[5]]
y = [0,0,1,1,1]
 
model = LogisticRegression()
model.fit(X,y)
hours = float(input('enter study hours:'))
prdicted_value = model.predict([[hours]])[0]
if prdicted_value == 0:
    print(f'Based on your study hour:{hours}, you may not pass the exam')
else:
    print(f'Based on your study hour:{hours}, you may pass the exam')