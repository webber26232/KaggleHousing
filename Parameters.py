import pandas as pd
import numpy as np
from TransformGridSearchCV import transform_grid_search_cv

X = pd.read_csv('X.csv')
y = pd.Series(np.ravel(pd.read_csv('y.csv')))
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import AdaBoostClassifier	
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#clf = DecisionTreeClassifier(criterion='gini')		
#clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=5,min_samples_split=200))
clf = XGBClassifier(objective='softprob',subsample=0.6,colsample_bytree=0.6,gamma=0.2,reg_lambda=10)
from MyCatTrans import CategoricalTransformer
transformer = CategoricalTransformer('manager_id',[0,1],18)

if __name__=='__main__':


	#out = transform_grid_search_cv(clf,tmp[features_to_use+['manager_id']],y,
	#{'n_estimators':[50,100,200,400,800]},n_jobs=-1,cv=None,data_transformer=transformer)
	#out.to_csv('out_Ada.csv',index=False)
	#out = transform_grid_search_cv(clf,X,y,
	#{'max_depth':[5,6,7,8,9],'min_samples_split':[50,100,150,200,250]},n_jobs=2,cv=None,data_transformer=transformer)
	#out.to_csv('out_Tree_2.csv',index=False)
	
	'''	
	out = transform_grid_search_cv(clf,tmp[features_to_use+['manager_id','area_code','building_code']],y,
	{'n_estimators':[125,250,500,1000],'max_depth':[3,4,5,6,7,8],'min_child_weight':[1,3,5],
	'subsample':[1,0.8,0.6],'colsample_bytree':[1,0.8,0.6]},'gamma':[0,0.1,0.2],'reg_lambda':[0,0.1,1,10,100]},
	n_jobs=1,cv=None,data_transformer=transformer)
	out = transform_grid_search_cv(clf,tmp[features_to_use+['manager_id','building_code','area_code']],y,
	{'gamma':[0,0.1,0.2],'min_child_weight':[1,3,5],'reg_lambda':[0,0.1,1,10,100]},n_jobs=1,cv=None,data_transformer=transformer)'''
	out = transform_grid_search_cv(clf,X,y,{'n_estimators':[500,1000,1500],'max_depth':[6,7,8],'min_child_weight':[2,3,4]},n_jobs=1,cv=None,data_transformer=transformer)
	out.to_csv('out_XGB2.csv',index=False)
