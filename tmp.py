import pandas as pd
import numpy as np
train = pd.read_json('train.json')
#test = pd.read_json('test.json')
#df = pd.concat([train,test],axis=0)
#del train, test
df =train.reset_index()
del train

df.price = np.log(df.price)
original_addresses = df.street_address.str.replace('\(.*?\)','').str.replace('&',' and ').str.strip().str.lower().str.replace(' +',' ')

adrs = pd.read_csv('formatted_adrs_nodup.csv',encoding='gbk')

adrs.original_adrs.fillna('',inplace=True)
adrs['building'] = adrs['building'] =  adrs.street_number.fillna('').str[:] + ' ' + adrs.premise.fillna('').str[:] + adrs.route.fillna('').str[:]

adrs['building'] = adrs['building'].where(adrs['building']!=' ',adrs['formatted_adrs'].str.replace(',.+','').str.replace(' #.+',''))

median = df[['latitude','longitude']].median()
mahathen_dist = (df[['latitude','longitude']] - median).abs().sum(axis=1)
new_lat = df.latitude.where(mahathen_dist<=mahathen_dist.quantile(0.995),original_addresses.map(adrs.set_index('original_adrs')['lat']))
new_lng = df.longitude.where(mahathen_dist<=mahathen_dist.quantile(0.995),original_addresses.map(adrs.set_index('original_adrs')['lng']))
df.latitude = df.latitude.where(new_lat.isnull(),new_lat)
df.longitude = df.longitude.where(new_lng.isnull(),new_lng)

df['formatted_adrs'] = original_addresses.map(adrs.set_index('original_adrs')['building']).fillna(original_addresses)


adrs['adrs_quality'] = adrs.premise.notnull()*1+adrs.route.notnull()*1+adrs.street_number.notnull()*1+adrs.subpremise.notnull()*1
df['adrs_quality'] = original_addresses.map(adrs.set_index('original_adrs')['adrs_quality']).fillna(-1)

adrs['street_name'] = (adrs.premise.fillna('').str[:] + adrs.route.fillna('').str[:]).fillna(adrs['formatted_adrs'].str.replace(',.+','').str.replace(' #.+',''))
street_name = original_addresses.map(adrs.set_index('original_adrs')['street_name']).fillna(original_addresses)

def encode_by_count(column):
    count = column.value_counts()
    return column.map(pd.Series(range(count.size),index=count.index))
df['street_code'] = encode_by_count(street_name)
del street_name, adrs, original_addresses, median, mahathen_dist


def create_area_code(frame,cut_size,target_column_prefix,method='width',keep='code'):
    if target_column_prefix+'_code' in frame.columns:
        frame = frame.drop(target_column_prefix+'_code',axis=1)
    if target_column_prefix+'_count' in frame.columns:
        frame = frame.drop(target_column_prefix+'_count',axis=1)
    if method == 'width':
        frame['lat_group'] = pd.cut(frame.latitude,bins=cut_size)
        frame['lng_group'] = pd.cut(frame.longitude,bins=cut_size)
    elif method == 'depth':
        quantile_list = [float(x)/cut_size for x in range(cut_size+1)]
        frame['lat_group'] = pd.qcut(frame.latitude,quantile_list)
        frame['lng_group'] = pd.qcut(frame.longitude,quantile_list)
    else:
        raise ValueError('method should be width or depth')
    area = frame.groupby(['lat_group','lng_group']).size().sort_values().reset_index().reset_index().set_index(['lat_group','lng_group']).rename(columns={'index':target_column_prefix+'_code',0:target_column_prefix+'_count'})
    if keep == 'code':
        return frame.merge(area,how='left',left_on=['lat_group','lng_group'],right_index=True).drop(['lat_group','lng_group',target_column_prefix+'_count'],axis=1)
    elif keep == 'count':
        return frame.merge(area,how='left',left_on=['lat_group','lng_group'],right_index=True).drop(['lat_group','lng_group',target_column_prefix+'_code'],axis=1)
    elif keep == 'both':
        return frame.merge(area,how='left',left_on=['lat_group','lng_group'],right_index=True).drop(['lat_group','lng_group'],axis=1)
    else:
        raise ValueError('keep should be code or count or both')
	
'''df = create_area_code(df,400,'area_large')
df = create_area_code(df,1200,'area_medium')
df = create_area_code(df,2000,'area_small')
df = create_area_code(df,20,'area_qlarge','depth')
df = create_area_code(df,40,'area_qmedium','depth')
df = create_area_code(df,60,'area_qsmall','depth')'''
df = create_area_code(df,50,'area_qsmall','depth')

df['building_code'] = encode_by_count(df.building_id)
df['adrs_code'] = encode_by_count(df.formatted_adrs)

adrs_features = ['adrs_quality']

df['num_photos'] = df['photos'].apply(len)
df['num_features'] = df['features'].apply(len)
df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))

df['created'] = pd.to_datetime(df['created'])
df['created_month'] = df['created'].dt.month
df['created_day'] = df['created'].dt.day
df['created_day_of_year'] = df['created'].dt.dayofyear
df['created_day_of_week'] = df['created'].dt.dayofweek
df['created_tens_of_month'] = df['created_day'].apply(lambda x:int(str(x).zfill(2)[0])).replace(3,2)
df['created_hour'] = df['created'].dt.hour
df['created_quarter_of_day'] = (df['created_hour']/6).astype(int)

df.bedrooms = df.bedrooms.replace(0,0.5)
df.bathrooms = df.bathrooms.replace(0,0.5)
df['rooms'] = df['bedrooms'] + df['bathrooms']
df['bed_bath_rate'] = df['bedrooms'] / df['bathrooms']
df['bed_bath_diff'] = df['bedrooms'] - df['bathrooms']

df['price_per_bed'] = df['price'] / df['bedrooms']
df['price_per_bath'] = df['price'] / df['bathrooms']
df['price_per_room'] = df['price'] / df['rooms']



from geopy import distance
time_square = [40.757888,-73.985613]
center_park = [40.7789,-73.968384]
world_trade_center = [40.711522,-74.013173]
brooklyn_center = [40.714470,-73.961303]
airport = [40.644953, -73.787114]
king_theater = [40.645593, -73.958087]
df['dist_to_tmsq'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,time_square).meters,axis=1)
df['dist_to_ctpk'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,center_park).meters,axis=1)
df['dist_to_wtc'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,world_trade_center).meters,axis=1)
df['dist_to_bkc'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,brooklyn_center).meters,axis=1)
df['dist_to_ap'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,airport).meters,axis=1)
df['dist_to_kt'] = df[['latitude','longitude']].apply(lambda x:distance.vincenty(x,king_theater).meters,axis=1)
distance_features = ['dist_to_tmsq','dist_to_ctpk','dist_to_wtc','dist_to_bkc','dist_to_ap','dist_to_kt']




features_to_use = ['bathrooms', 'bedrooms', 'rooms', 'latitude', 'longitude', 'price',
                   'num_photos', 'num_features', 'num_description_words',
                   'created_hour','created_quarter_of_day','created_day','created_day_of_week','created_day_of_year', 'created_month','created_tens_of_month',
                   'bed_bath_rate','bed_bath_diff',
                   'price_per_bath','price_per_bed','price_per_room']


category_features = ['building_code','street_code','adrs_code','area_qsmall_code']

features = features_to_use + category_features + adrs_features + distance_features
X = df[features+['manager_id']]

y = df[df.interest_level.notnull()]['interest_level'].map({'high':0,'medium':1,'low':2})

#X.to_csv('simple_X.csv',index=False)
del df

#y.to_csv('y.csv',index=False)

import time
from sklearn.metrics import log_loss
from MyCatTrans import CategoricalTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

def cv_log_loss(X,y):
    start_time = time.time()
    test_index = np.array([],dtype=int)
    prediction_encode = np.zeros((0,3))
    for train,test in StratifiedKFold(10,shuffle=True,random_state=123).split(X,y):
        transformer = CategoricalTransformer('manager_id',[0,1],18)
        clf = RandomForestClassifier(n_estimators=666,random_state=123,min_samples_leaf=7,n_jobs=-1)

        test_index = np.append(test_index,test)
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        X_train = transformer.fit_transform(X_train,y_train)
        X_test = transformer.transform(X_test)  

        clf.fit(X_train,y_train)
        prediction_encode = np.concatenate([prediction_encode,clf.predict_proba(X_test)])

    inverted_index = np.zeros(y.size,dtype=int)
    inverted_index[test_index] = np.arange(y.size,dtype=int)
    duration = time.time() - start_time
    hours = int(duration/3600)
    mins = int(duration % 3600 / 60)
    seconds = int(duration % 60)
    print('Test finished in {0} hours {1} mins {2} seconds'.format(hours,mins,seconds))
    return log_loss(y,prediction_encode[inverted_index])

def add_feature(df,cat,met,method):
    if method == 'rank':
        return df.groupby(cat)[met].rank(method='average',pct=True,ascending=False)
    elif method == 'diff':
        return df[met] - df[cat].map(df.groupby(cat)[met].median())
    elif method == 'median':
        return df[cat].map(df.groupby(cat)[met].median())
    elif method == 'std':
        return df[cat].map(df.groupby(cat)[met].std()).fillna(0)
    
    
    
room_features = ['bedrooms','rooms','bathrooms']#bed_bath
adrs_features = ['area_qsmall_code','adrs_code','building_code','street_code'] #area_small_code

date_features = ['created_month','created_day_of_week']#independent
date_features_in_month = ['created_tens_of_month','created_day']
date_features_in_day = ['created_quarter_of_day','created_hour']

#numerical feature groups
price_features = ['price','price_per_bed','price_per_room','price_per_bath']

describe_features = ['num_features','num_description_words',                                      'num_photos']
distance_features = ['dist_to_tmsq','dist_to_wtc','dist_to_bkc','dist_to_ap','dist_to_kt',      'dist_to_ctpk']

bed_bath_features = ['bed_bath_rate','bed_bath_diff']
other_features = ['created_day_of_year','adrs_quality']#independent

#measure methods
deviation_measurement = ['rank','diff']
group_feature_measurement = ['median','std']#independent


kept_features =  ['price_rank_area_qsmall_code_code',
'price_rank_bedrooms',
'price_rank_rooms',
'price_per_bed_rank_bedrooms',
'price_per_bed_rank_rooms',
'num_features_rank_area_qsmall_code',
'num_features_rank_bedrooms',
'num_features_rank_rooms',
'num_photos_rank_created_month',
'dist_to_tmsq_rank_bedrooms',
'dist_to_tmsq_rank_rooms',
'dist_to_tmsq_rank_created_tens_of_month',
'dist_to_tmsq_rank_created_hour',
'dist_to_ctpk_rank_created_month',
'dist_to_wtc_rank_created_month',
'dist_to_ap_rank_created_tens_of_month',
'dist_to_kt_rank_created_day',
'adrs_quality_rank_building_code',
'adrs_quality_rank_bedrooms',
'adrs_quality_rank_rooms',
'adrs_quality_rank_bathrooms']
best_score = 1
feature_size = X.shape[1]
for method in deviation_measurement + group_feature_measurement:
    for met in price_features + describe_features + distance_features + bed_bath_features + other_features:
        for cat in adrs_features + room_features + date_features: 
            name = met + '_' + method + '_' + cat
            if name in kept_features:
                new_feature = add_feature(X,cat,met,method)
                new_feature.name = name
                X = pd.concat([X,new_feature],axis=1)
            if method == 'rank':
                continue
            if cat in ['area_qsmall_code','adrs_code','building_code'] and met in distance_features:
                continue
            new_feature = add_feature(X,cat,met,method)
            new_feature.name = name
            print('Start working on',new_feature.name)
            score = cv_log_loss(pd.concat([X,new_feature],axis=1),y)
            if score <= best_score:
                best_score = score
                X = pd.concat([X,new_feature],axis=1)
                print(score)
                print(X.shape[1] - feature_size,'features added')
            print('')
                

X.columns.to_series().to_csv('kept_features.csv',index=False)