import pandas as pd
import numpy as np
train = pd.read_json('train.json')
#test = pd.read_json('test.json')
#df = pd.concat([train,test],axis=0)
#del train, test
df =train.reset_index()
del train

#drop boston building
df.price = np.log(df.price)
original_addresses = df.street_address.str.replace('\(.*?\)','').str.replace('&',' and ').str.strip().str.lower().str.replace(' +',' ')

adrs = pd.read_csv('formatted_adrs_nodup.csv',encoding='gbk')

adrs.original_adrs.fillna('',inplace=True)
adrs['building'] = adrs['building'] =  adrs.street_number.fillna('').str[:] + ' ' + adrs.premise.fillna('').str[:] + adrs.route.fillna('').str[:]

adrs['building'] = adrs['building'].where(adrs['building']!=' ',adrs['formatted_adrs'].str.replace(',.+','').str.replace(' #.+',''))

median = df[['latitude','longitude']].median()
mahathen_dist = (df[['latitude','longitude']] - median).abs().sum(axis=1)
#df['bad_codint_quality'] = (mahathen_dist>mahathen_dist.quantile(0.995))*1
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

#df[['listing_id','formatted_adrs','latitude','longitude']].to_csv('cood.csv',index=False)

ifm_dict = {'convenience_store':[500],'home_goods_store':[500],'department_store':[2000],
'bar':[500],'cafe':[500],'restaurant':[300],
'train_station':[3000],'bus_station':[1000,300],'subway_station':[2000,500],
'laundry':[1000],'bank':[1000],'pharmacy':[1000],'church':[1000],'school':[500]}


ambs_features = [ifm+'_'+str(x) for ifm in ifm_dict for x in ifm_dict[ifm]]
ambs = pd.read_csv('cood_with_ambs_patch.csv',encoding='gbk')

ambs['transportation_sum'] = ambs[['subway_station_500','bus_station_300']].sum(axis=1)
ambs['food_sum'] = ambs[['bar_500','cafe_500','restaurant_300']].sum(axis=1)
ambs['convinience_sum'] = ambs[['laundry_1000','bank_1000','pharmacy_1000','church_1000','school_500']].sum(axis=1)
ambs['store_sum'] = ambs[['home_goods_store_500','convenience_store_500']].sum(axis=1)
ambs['ambs_total'] = ambs[['food_sum','convinience_sum','store_sum','subway_station_2000','bus_station_1000']].sum(axis=1)

ambs_features += ['food_sum','convinience_sum','transportation_sum','store_sum','ambs_total']

df = df.merge(ambs[ambs_features+['listing_id']],how='left',left_on='listing_id',right_on='listing_id')
del ambs, ifm_dict

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
	
df = create_area_code(df,400,'area_large')
df = create_area_code(df,1200,'area_medium')
df = create_area_code(df,2000,'area_small')
df = create_area_code(df,20,'area_qlarge','depth')
df = create_area_code(df,40,'area_qmedium','depth')
df = create_area_code(df,60,'area_qsmall','depth')

df['building_code'] = encode_by_count(df.building_id)
df['adrs_code'] = encode_by_count(df.formatted_adrs)

#adrs_features = ['bad_codint_quality','adrs_quality']
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
                   'created_day_of_year', 'created_month', 'created_day_of_week', 'created_day','created_tens_of_month',
                   'bed_bath_rate','bed_bath_diff',
                   'price_per_bath','price_per_bed','price_per_room']


category_features = ['building_code','street_code','adrs_code',
                     'area_large_code','area_medium_code','area_small_code',
                     'area_qlarge_code','area_qmedium_code','area_qsmall_code']

					 
metrics = ['price','price_per_bed','price_per_bath','price_per_room',
           'bed_bath_rate','bed_bath_diff',
           'num_features','num_photos','num_description_words',
           'created_day_of_year'] + ambs_features + distance_features + adrs_features
		   

cat_features = []
for cat in category_features + ['created_month','created_day','created_day_of_week', 'created_tens_of_month','bedrooms','bathrooms','rooms']:
    for met in metrics:
        if (met in ambs_features  or met in distance_features) and 'code' in cat:
            continue
        #[cat+'_mean_'+met,'diff_from_'+cat+'_mean_'+met]
        #df[cat+'_mean_'+met] = df[cat].map(df.groupby(cat)[met].mean())
        #df['diff_from_'+cat+'_mean_'+met] = df[met] - df[cat+'_mean_'+met]

        cat_features.extend([cat+'_median_'+met,cat+'_std_'+met,'diff_from_'+cat+'_median_'+met,'rank_of_'+cat+'_'+met])
        gb_object = df.groupby(cat)[met]
        df[cat+'_median_'+met] = df[cat].map(gb_object.median())  
        df[cat+'_std_'+met] = df[cat].map(gb_object.std()).fillna(0)
        df['diff_from_'+cat+'_median_'+met] = df[met] - df[cat+'_median_'+met]
        df['rank_of_'+cat+'_'+met] = gb_object.rank(method='average',pct=True,ascending=False)
		
#cat_features = [x for x in cat_features if not x.startswith('diff')] + [x for x in cat_features if x.startswith('diff') and not x.endswith('bed_bath_rate')]


num_cut_features = []

for met_to_cut in metrics:
    if met_to_cut.endswith('00'):
        continue
    if 'quality' in met_to_cut:
        continue
    #large_group = pd.cut(df[met_to_cut],bins=5)
    #small_group = pd.cut(df[met_to_cut],bins=10)
    if met_to_cut in distance_features:
        qlarge_group = pd.qcut(df[met_to_cut],[float(x)/10 for x in range(11)])
    else:
        qlarge_group = pd.qcut(df[met_to_cut],[float(x)/5 for x in range(6)])
    #qsmall_group = pd.qcut(df[met_to_cut],[float(x)/10 for x in range(11)])
    for met_to_cal in metrics:
        num_cut_features.extend([met_to_cut+'_group_median_'+met_to_cal,met_to_cut+'_group_std_'+met_to_cal,'diff_from_'+met_to_cut+'_group_median_'+met_to_cal,'rank_of_'+met_to_cut+'_group_'+met_to_cal])
        gb_object = df.groupby(qlarge_group)[met_to_cal]
        df[met_to_cut+'_group_median_'+met_to_cal] = gb_object.transform(lambda x:x.median())
        df[met_to_cut+'_group_std_'+met_to_cal] = gb_object.transform(lambda x:x.std()).fillna(0)
        df['diff_from_'+met_to_cut+'_group_median_'+met_to_cal] = df[met_to_cal] - df[met_to_cut+'_group_median_'+met_to_cal]
        df['rank_of_'+met_to_cut+'_group_'+met_to_cal] = gb_object.rank(method='average',pct=True,ascending=False)
        



features = list(set(features_to_use + cat_features + num_cut_features + category_features + adrs_features + ambs_features + distance_features))
X = df[features+['manager_id']]

y = df[df.interest_level.notnull()]['interest_level'].map({'high':0,'medium':1,'low':2})

X.to_csv('X.csv',index=False)
y.to_csv('y.csv',index=False)
