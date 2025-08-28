from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

y_true = [1,0,1,1,0,1,0]
y_pred =[1,0,1,0,0,1,1,]
print('Accuracy score:', accuracy_score(y_true,y_pred))
print('precision score:', precision_score(y_true,y_pred))
print('recall score:', recall_score(y_true,y_pred))
print('f1 score:', f1_score(y_true,y_pred))       