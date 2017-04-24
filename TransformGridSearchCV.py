import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
#from sklearn.base import TransformerMixin
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.externals import six
from collections import Sequence
from sklearn.base import clone
from sklearn.metrics import log_loss
import time

def transform_grid_search_cv(estimator,X,y,param_grid,n_jobs=1,cv = None,data_transformer=None):
    test_index = np.array([],dtype=int)
    _check_param_grid(param_grid)
    params = [x for x in ParameterGrid(param_grid)]
    l = [np.zeros(shape=(0,len(set(y)))) for _ in range(len(params))]
    if cv is None:
        cv=StratifiedKFold(10,shuffle=True,random_state=123)
    count = 0
    for train,test in cv.split(X,y):
        start_time = time.time()
        test_index = np.append(test_index,test)
        if isinstance(X,pd.DataFrame):
            X_train = X.iloc[train]
            X_test = X.iloc[test]
            y_train = y.iloc[train]
        else:
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]            
        if data_transformer is not None:
            X_train = data_transformer.fit_transform(X_train,y_train)
            X_test = data_transformer.transform(X_test)
        if n_jobs == 1:
            out = [_fit_predict(clone(estimator),X_train,X_test,y_train,parameters) for parameters in params]
        else:
            out = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(clone(estimator),X_train,X_test,y_train,parameters) for parameters in params)
        for i in range(len(l)):
            l[i] = np.concatenate([l[i],out[i]])
            count += 1
        duration = time.time() - start_time
        hours = int(duration/3600)
        mins = int(duration % 3600 / 60)
        seconds = duration % 60
        print('fold {0} finished in {1} hours {2} mins {3} seconds'.format(count,hours,mins,seconds))
    inverted_index = np.zeros(y.size,dtype=int)
    inverted_index[test_index] = np.arange(y.size,dtype=int)
    for i in range(len(l)):
        params[i]['score'] = log_loss(y,l[i][inverted_index])
    return pd.DataFrame(params)

def _fit_predict(estimator,X_train,X_test,y_train,parameters):
    estimator.set_params(**parameters)
    print(parameters,' start fitting')
    return estimator.fit(X_train,y_train).predict_proba(X_test)
	

def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, six.string_types) or
                    not isinstance(v, (np.ndarray, Sequence))):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))
