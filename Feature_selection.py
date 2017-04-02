import pandas as pd
import numpy as np

X = pd.read_csv('X.csv').drop('manager_id',axis=1)
y = pd.Series(np.ravel(pd.read_csv('y.csv',header=None)))
print(X.shape)

#from xgboost import XGBClassifier
#clf = XGBClassifier(objective='softprob',n_estimators=500,max_depth=5,subsample=0.6,colsample_bytree=1,gamma=0.2,min_child_weight=3,reg_lambda=10)

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=666,n_jobs=-1)

'''
clf.fit(X,y)
pd.concat([pd.Series(clf.feature_importances_,name = 'importance'),pd.Series(X.columns,name='columne_name')],axis=1).to_csv('RF_feature_importance.csv',index=False)'''



from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
rfecv = RFECV(clf,cv=StratifiedKFold(10,shuffle=True,random_state=123),scoring='neg_log_loss',step=20,verbose=10)
rfecv.fit(X,y)
pd.DataFrame([X.columns,rfecv.support_,rfecv.ranking_]).T.to_csv('feature_selection_on_manual.csv')
pd.DataFrame([range(1, len(rfecv.grid_scores_) + 1),rfecv.grid_scores_]).T.to_csv('feature_scores.csv')


'''
import nltk
text = df.description.str[:]+' '+df.features.apply(lambda x:'. '.join(x)).str[:]
tmp = text.str.replace('<[^>]+?>',' ').str.replace('w/',' ').str.replace('<a +website_redacted','').str.replace('[^a-zA-Z0-9,. ]+',' ').str.replace('\s{2,}','. ').str.replace('\.+','.').str.replace('^. ','').str.lower()'''

