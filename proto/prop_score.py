import pandas
import numpy
print "read 1"
dev_train = pandas.read_csv('../data/dev_train_basic.csv')
cookie_basic = pandas.read_csv('../data/cookie_all_basic.csv')

cookie_basic = cookie_basic[['drawbridge_handle', 'cookie_id']]
dev_train = dev_train[['drawbridge_handle', 'device_id']]
print "union"
mst_user = pandas.DataFrame(numpy.r_[dev_train, cookie_basic], columns=["id", "device_id"])
print "read 2"
id_all_ip  = pandas.read_csv('../data/id_all_property_flat.csv', header=None, names=range(4))
print "merge"
df_ip = id_all_ip.merge(mst_user, how='left', left_on=0, right_on='device_id')

df_ip = df_ip.ix[:, 2:].dropna()
print "groupby"
df_ip = df_ip.groupby(2).agg(pandas.Series.nunique)
print "score"
df_ip = df_ip[df_ip['device_id'] > 1]
df_ip['score'] = (df_ip['device_id'] - df_ip['id']) / (df_ip['device_id'] - 1)
print "output"
df_ip.to_csv('prop_score.csv')
