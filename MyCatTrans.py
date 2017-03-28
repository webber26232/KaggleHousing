from sklearn.base import BaseEstimator, TransformerMixin
from collections import Iterable
import pandas as pd
import numpy as np

class CategoricalTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, target_column, label_to_use=None, threshold='mean',coeficient=1,alpha=0.01):
		if isinstance(target_column,Iterable) and not isinstance(target_column,str):
			self.target_column = [x for x in target_column]
		else:
			self.target_column = target_column
		self.label_to_use = label_to_use
		self.threshold = threshold
		self.coeficient = coeficient
		self.alpha = alpha
		self.pre_fix = ''
		
		if isinstance(self.target_column,Iterable) and not isinstance(self.target_column,str):
			for i in range(len(self.target_column)):
				if isinstance(self.target_column[i],str) or isinstance(self.target_column[i],int) or isinstance(self.target_column[i],float):
					self.pre_fix += (str(self.target_column[i]) + '_')
				elif isinstance(self.target_column[i],pd.Series):
					self.pre_fix += (self.target_column[i].name + '_')
				else:
					self.pre_fix += (str(i) + '+')
		else:
				if isinstance(self.target_column,str) or isinstance(self.target_column,int) or isinstance(self.target_column,float):
					self.pre_fix += (str(self.target_column) + '_')
				elif isinstance(self.target_column,pd.Series):
					self.pre_fix += (self.target_column.name + '_')
				else:
					self.pre_fix += (str(0) + '+')


	def _reset(self):

		if hasattr(self, 'mapping_'):
			self.mapping_ = None
			self.global_ratio = None

	def fit(self, X,y):
		if not isinstance(y,pd.Series):
			y = pd.Series(y)
		
		if X.shape[0] != y.size:
			raise ValueError('X and y must have same size')
		if not isinstance(X,pd.DataFrame):
			raise TypeError('X has to be a pandas DataFrame')
			
		self._reset()
		
		if self.label_to_use is not None:
			extra_labels = set(self.label_to_use) - set(y)
			if len(extra_labels)>0:
				print('Runtime Warning: Output contains labels out of y: {0}'.format(list(extra_labels)))
		else:
			self.label_to_use = list(set(y))
			extra_labels = set([])
			

		
		if isinstance(self.target_column,Iterable) and not isinstance(self.target_column,str):
			groups = X.groupby([y]+self.target_column).size()
		else:
			groups = X.groupby([y]+[self.target_column]).size()
		tmp = groups.unstack(level=0,fill_value=0)
		
		for additional_label in extra_labels:
			tmp[additional_label] = 0
			
		record_count = tmp.sum(axis=1)
		self.global_ratio = tmp.sum(axis=0) / y.size
		for additional_label in extra_labels:
			self.global_ratio[additional_label] = 0		
		
		if isinstance(self.threshold,str):
			if self.threshold == 'median':
				threshold_value = record_count.median()
			elif self.threshold == 'mean':
				threshold_value = y.size/tmp.index.size
		elif isinstance(self.threshold,int):
			threshold_value = self.threshold
		elif isinstance(self.threshold,float):
			threshold_value = record_count.quantile(self.threshold)
		lambda_value =  1.0 / (1.0 + np.exp((threshold_value-record_count)*self.coeficient))	

		for label in self.label_to_use:
			ratio = tmp[label]/record_count
			weight = (ratio * lambda_value + (1 - lambda_value) * self.global_ratio[label])
			randoms = (1 + (np.random.uniform(size=tmp.shape[0])-0.5)*self.alpha)
			tmp[self.pre_fix+str(label)] = weight * randoms
		self.mapping_ = tmp[[self.pre_fix+str(label) for label in self.label_to_use]]
		
		return self

	def transform(self, X, inplace=True):
		
		X = X.merge(self.mapping_, how = 'left', left_on = self.target_column, right_index=True)
		
		for label in self.label_to_use:
			X[self.pre_fix+str(label)].fillna(self.global_ratio[label],inplace=True)

		if inplace:
			if isinstance(self.target_column,Iterable) and not isinstance(self.target_column,str):
				for column in self.target_column:
					column_name = ''
					if isinstance(column,str) or isinstance(column,int) or isinstance(column,float):
						column_name = column
					elif isinstance(column,pd.Series):
						column_name = column.name
					else:
						continue
					if column_name in X.columns:
						X.drop(column_name,axis=1,inplace=True)
			elif isinstance(self.target_column,str) or isinstance(self.target_column,int) or isinstance(self.target_column,float):
				if self.target_column in X.columns:
					X.drop(self.target_column,axis=1,inplace=True)
		return X
