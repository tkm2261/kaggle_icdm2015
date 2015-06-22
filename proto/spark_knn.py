import os
import numpy
import pandas
import scipy.sparse as spMat
import sys
import cPickle as pickle
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from multiprocessing import Pool
RECOMMEND_LIB_PATH = os.path.dirname(os.path.abspath(__file__))+"/../recommendation_survey/"
sys.path.append(RECOMMEND_LIB_PATH)

from recommend.io import RecommenderData
from recommend.svd import SVD
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest

import logging
log_fmt = '%(asctime)s %(name)s [%(levelname)s]%(funcName)s %(message)s'
logging.basicConfig(format=log_fmt,
                    filemode='w',
                    level=logging.DEBUG)

from sklearn.cluster import MiniBatchKMeans
from logging import getLogger
logger = getLogger(__name__)
#from pyspark import SparkContext, StorageLevel
#sc = SparkContext('local[16]', 'test')

if __name__ == '__main__':
    logger.info('load data')
    with open('recommend_data.pkl') as f:
        recommend_data = pickle.load(f)
    with open('result.svd') as f:
        svd = pickle.load(f)

    logger.info('start lsh')

    map_idx2user = recommend_data.map_idx2user
    nbrs_num = 2

    knn = LSHForest(n_estimators=20,
                    min_hash_match=4,
                    n_candidates=nbrs_num,
                    n_neighbors=nbrs_num).fit(svd.user_matrix)

    logger.info('predict')


    def aaa(args):
        user_idx, vec = args
        list_ret = []
        if user_idx % 1000 == 0:
            print user_idx
        distancess, indicess = knn.kneighbors([vec],
                                            n_neighbors=2)
        distances = distancess[0]
        indices = indicess[0]

        user_id = map_idx2user[user_idx]
            
        for i, n_user_idx in enumerate(indices):
            if user_idx == n_user_idx:
                continue
            n_user_id = recommend_data.map_idx2user[n_user_idx]
            return (user_id, n_user_id, distances[i])


    indexed_matrix = [(i , svd.user_matrix[i]) 
                      for i in xrange(svd.user_matrix.shape[0])]

    p = Pool()
    #map(aaa, indexed_matrix)
    result = p.map_async(aaa, indexed_matrix, chunksize=10000)
    p.close()
    p.join()

    pandas.DataFrame(result).to_csv('dist.csv', indexFalse)
