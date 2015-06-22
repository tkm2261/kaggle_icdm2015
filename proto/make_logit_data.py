import pandas
import numpy

dev_train = pandas.read_csv('../data/dev_train_basic.csv')
cookie_basic = pandas.read_csv('../data/cookie_all_basic.csv')

cookie_basic = cookie_basic[['drawbridge_handle', 'cookie_id']]
dev_train = dev_train[['drawbridge_handle', 'device_id']]
print "union"
data = numpy.r_[dev_train, cookie_basic].T
mst_user = pandas.DataFrame(data[0],
                            columns=["id"], index=data[1])

print "start"
out = open('data_logit.csv', 'wb')

positive_cnt = 0
with open('list_nearst.csv') as f:
    cnt = 0
    for line in f:
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, positive_cnt
        line = line.strip().split(',')
        id1 = line[0]
        id2 = line[1]
        uid1 = mst_user.ix[id1, 'id']
        uid2 = mst_user.ix[id2, 'id']
        dist = float(line[2])
        flag = 0
        
        if uid1 == uid2 and uid1 != '-1':
            flag = 1
            positive_cnt += 1
        out.write('%s,%s,%s,%s,%s,%s\n'%(flag,
                                         dist,
                                         id1,
                                         id2,
                                         uid1,
                                         uid2
                                         ))
    out.close()
