
def flat_line(line):
    line = line.strip().split(',')
    #dev_id = line[0]
    #is_cookie = bool(line[1])
    prefix = ','.join(line[:1])

    list_ip = ','.join(line[1:])
    list_ip = list_ip.strip('(){}').split('),(')
    if line[0] == 'device_or_cookie_id':
        return []

    return ['%s,%s'%(prefix, ip_ele)for ip_ele in list_ip]

if __name__ == '__main__':

    import time
    t = time.time()

    """
    from pyspark import SparkContext, StorageLevel
    sc = SparkContext('local[16]', 'test')
    data = sc.textFile('../data/id_all_property.csv')

    data = data.flatMap(flat_line)
    data.saveAsTextFile('../data/id_all_property_flat')
    """
    out = open('../data/property_category_flatten.csv', 'w')
    with open('../data/property_category.csv') as f:
        f.readline()
        for line in f:
            for aaa in flat_line(line):
                out.write('%s\n'%aaa)
    out.close()


    print time.time() - t 
