import pandas as pd
import numpy as np
train = pd.read_json('train.json')
#test = pd.read_json('test.json')
#df = pd.concat([train,test],axis=0)
df =train

#drop boston building
df = df.drop(df[df.listing_id==7202226].index)
df.price = np.log(df.price)
original_addresses = df.street_address.str.replace('\(.*?\)','').str.replace('&',' and ').str.strip().str.lower().str.replace(' +',' ')

adrs = pd.read_csv('formatted_adrs_nodup.csv',encoding='gbk')

adrs.original_adrs.fillna('',inplace=True)
adrs['building'] = adrs['building'] =  adrs.street_number.fillna('').str[:] + ' ' + adrs.premise.fillna('').str[:] + adrs.route.fillna('').str[:]

adrs['building'] = adrs['building'].where(adrs['building']!=' ',adrs['formatted_adrs'].str.replace(',.+','').str.replace(' #.+',''))

median = df[['latitude','longitude']].median()
mahathen_dist = (df[['latitude','longitude']] - median).abs().sum(axis=1)
df['bad_codint_quality'] = (mahathen_dist>mahathen_dist.quantile(0.995))*1
new_lat = df.latitude.where(mahathen_dist<=mahathen_dist.quantile(0.995),original_addresses.map(adrs[adrs.postal_code.notnull()].set_index('original_adrs')['lat']))
new_lng = df.longitude.where(mahathen_dist<=mahathen_dist.quantile(0.995),original_addresses.map(adrs[adrs.postal_code.notnull()].set_index('original_adrs')['lng']))
df.latitude = df.latitude.where(new_lat.isnull(),new_lat)
df.longitude = df.longitude.where(new_lng.isnull(),new_lng)

df['formatted_adrs'] = original_addresses.map(adrs.set_index('original_adrs')['building']).fillna(original_addresses)


adrs['adrs_quality'] = adrs.premise.notnull()*1+adrs.route.notnull()*1+adrs.street_number.notnull()*1+adrs.subpremise.notnull()*1
df['adrs_quality'] = original_addresses.map(adrs.set_index('original_adrs')['adrs_quality']).fillna(-1)

adrs['street_name'] = (adrs.premise.fillna('').str[:] + adrs.route.fillna('').str[:]).fillna(adrs['formatted_adrs'].str.replace(',.+','').str.replace(' #.+',''))
street_name = original_addresses.map(adrs.set_index('original_adrs')['street_name']).fillna(original_addresses)
street_count = street_name.value_counts()
df['street_code'] = street_name.map(pd.Series(range(street_count.size),index=street_count.index))
del street_name, street_count, adrs, original_addresses, median, mahathen_dist

#df[['listing_id','formatted_adrs','latitude','longitude']].to_csv('cood.csv',index=False)
ifm_dict = {'convenience_store':[500],
'home_goods_store':[500],
'department_store':[2000],
'bar':[500],'cafe':[500],'restaurant':[300],
'train_station':[3000],'bus_station':[1000,300],
'subway_station':[2000,500],
'laundry':[1000],'bank':[1000],'pharmacy':[1000],'church':[1000],'school':[500]}

ambs_features = [ifm+'_'+str(x) for ifm in ifm_dict for x in ifm_dict[ifm]]
ambs = pd.read_csv('cood_with_ambs_patch.csv',encoding='gbk')

df = df.merge(ambs[ambs_features+['listing_id']],how='left',left_on='listing_id',right_on='listing_id')
del ambs, ifm_dict

def create_area_code(frame,cut_size,target_column_name,method='width'):
	if method == 'width':
		frame['lat_group'] = pd.cut(frame.latitude,bins=cut_size)
		frame['lng_group'] = pd.cut(frame.longitude,bins=cut_size)
	elif method == 'depth':
		quantile_list = [float(x)/cut_size for x in range(cut_size+1)]
		frame['lat_group'] = pd.qcut(frame.latitude,quantile_list)
		frame['lng_group'] = pd.qcut(frame.longitude,quantile_list)
	area = frame.groupby(['lat_group','lng_group']).size().sort_values().reset_index().reset_index().set_index(['lat_group','lng_group']).rename(columns={'index':target_column_name,0:'area_count'})
	return frame.merge(area,how='left',left_on=['lat_group','lng_group'],right_index=True).drop(['lat_group','lng_group','area_count'],axis=1)
	
df = create_area_code(df,150,'area_code_large')
df = create_area_code(df,200,'area_code_medium')
df = create_area_code(df,250,'area_code_small')

building_count = df.building_id.value_counts()
df['building_code'] = df.building_id.map(pd.Series(range(building_count.size),index = building_count.index))
adrs_count = df.formatted_adrs.value_counts()
df['adrs_code'] = df.formatted_adrs.map(pd.Series(range(adrs_count.size),index = adrs_count.index))
del adrs_count,building_count

adrs_features = ['bad_codint_quality','adrs_quality']

df['num_photos'] = df['photos'].apply(len)
df['num_features'] = df['features'].apply(len)
df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))
df['created'] = pd.to_datetime(df['created'])
df['created_month'] = df['created'].dt.month
df['created_day'] = df['created'].dt.day
df['created_day_of_year'] = df['created'].dt.dayofyear

df.bedrooms = df.bedrooms.replace(0,0.8)
df.bathrooms = df.bathrooms.replace(0,0.8)
df['rooms'] = df['bedrooms'] + df['bathrooms']
df['bed_bath_rate'] = df['bedrooms'] / df['bathrooms']
df['bed_bath_diff'] = df['bedrooms'] - df['bathrooms']

df['price_per_bed'] = df['price'] / df['bedrooms']
df['price_per_bath'] = df['price'] / df['bathrooms']
df['price_per_room'] = df['price'] / df['rooms']


features_to_use = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
                   'num_photos', 'num_features', 'num_description_words',
                   'created_day_of_year', 'created_month', 'created_day',
                   'bed_bath_rate','bed_bath_diff',
                   'price_per_bath','price_per_bed','price_per_room',]


category_code_features = ['building_code','street_code','adrs_code','area_code_large','area_code_medium','area_code_small']
metrics = ['price','price_per_bed','price_per_bath','price_per_room',
           'bed_bath_rate','bed_bath_diff',
           'num_features','num_photos','num_description_words',
           'created_month','created_day','created_day_of_year']
new_features = []
for cat in category_code_features:
	for met in metrics:
		#[cat+'_mean_'+met,'diff_from_'+cat+'_mean_'+met]
		#df[cat+'_mean_'+met] = df[cat].map(df.groupby(cat)[met].mean())
		#df['diff_from_'+cat+'_mean_'+met] = df[met] - df[cat+'_mean_'+met]
		
		new_features.extend([cat+'_median_'+met,'diff_from_'+cat+'_median_'+met])
		df[cat+'_median_'+met] = df[cat].map(df.groupby(cat)[met].median())        
		df['diff_from_'+cat+'_median_'+met] = df[met] - df[cat+'_median_'+met]

		



from xgboost import XGBClassifier
clf = XGBClassifier(objective='softprob',n_estimators=500,max_depth=5,subsample=0.6,colsample_bytree=1,gamma=0.2,min_child_weight=3,reg_lambda=10)

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

rfecv = RFECV(clf,cv=StratifiedKFold(10,shuffle=True,random_state=123),scoring='neg_log_loss',step=3,verbose=5)



from geopy import distance
time_square = [40.757888,-73.985613]
center_park = [40.7789,-73.968384]
world_trade_center = [40.711522,-74.013173]
brooklyn_center = [40.714470,-73.961303]
df['dist_to_tmsq'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,time_square).meters,axis=1)
df['dist_to_ctpk'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,center_park).meters,axis=1)
df['dist_to_wtc'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,world_trade_center).meters,axis=1)
df['dist_to_bkc'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,brooklyn_center).meters,axis=1)
distance_features = ['dist_to_tmsq','dist_to_ctpk','dist_to_wtc','dist_to_bkc']
features = features_to_use + new_features + category_code_features + adrs_features + ambs_features + distance_features


from MyCatTrans import CategoricalTransformer
transformer = CategoricalTransformer('manager_id',[0,1],20)
X = df[features+['manager_id']]

y = df['interest_level'].map({'high':0,'medium':1,'low':2})
X = transformer.fit_transform(X,y)


rfecv.fit(X,y)
pd.DataFrame([X.columns,rfecv.support_,rfecv.ranking_]).to_csv('feature_selection.csv')
pd.DataFrame([range(1, len(rfecv.grid_scores_) + 1,3),rfecv.grid_scores_]).to_csv('feature_scores.csv')


'''
import nltk
text = df.description.str[:]+' '+df.features.apply(lambda x:'. '.join(x)).str[:]
tmp = text.str.replace('<[^>]+?>',' ').str.replace('w/',' ').str.replace('<a +website_redacted','').str.replace('[^a-zA-Z0-9,. ]+',' ').str.replace('\s{2,}','. ').str.replace('\.+','.').str.replace('^. ','').str.lower()'''

