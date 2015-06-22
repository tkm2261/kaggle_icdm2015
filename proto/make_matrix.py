# -*- coding: utf-8 -*-
import os
import numpy
import pandas
import scipy.sparse as spMat
import sys
import cPickle as pickle
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

RECOMMEND_LIB_PATH = os.path.dirname(os.path.abspath(__file__))+"/../recommendation_survey/"
sys.path.append(RECOMMEND_LIB_PATH)
import multiprocessing as mp
from recommend.io import RecommenderData
from recommend.svd import SVD
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest

import logging
log_fmt = '%(asctime)s %(name)s [%(levelname)s]%(funcName)s %(message)s'
logging.basicConfig(format=log_fmt,
                    filemode='w',
                    level=logging.DEBUG)
from logging import getLogger
logger = getLogger(__name__)

QUEUE_STOP_WORD = "==kill=="

def write_listener(queue, filepath):
    '''形態素解析ファイル出力リスナー関数
    '''

    f = open(filepath, 'ab')
    while 1:
        m = queue.get()
        if m == QUEUE_STOP_WORD:
            break
        f.write(m)
        f.flush()
    f.close()

def write_worker(tuple_data,
                 map_id,
                 queue):
    '''形態素解析ファイル出力ワーカー関数

    '''
    
    text = "%s,%s,%s\n"%(map_id[int(tuple_data[0])],
                         map_id[int(tuple_data[1])],
                         tuple_data[2])
    queue.put(text)


class MakeMatrix(object):
    
    def __init__(self,
                 data_paths,
                 score_paths,
                 data_prefix_col_num):

        self.data_paths = data_paths
        self.score_paths = score_paths
        self.data_prefix_col_num = data_prefix_col_num

        self.pd_active_ip = self._get_active_ip()

        self.row = []
        self.col = []
        self.data = []

        self.map_idx2user = {}
        self.map_idx2item = {}
        self.map_user2idx = {}
        self.map_item2idx = {}
        self.user_cnt = 0
        self.item_cnt = 0

    def _get_active_ip(self):
        df = pandas.read_csv(self.score_paths[0], index_col="2")

        for score_path in self.score_paths[1:]:
            df2 = pandas.read_csv(score_path, index_col="2")[df.columns]
            df2.columns = df.columns
            df = df.append(df2)

        df = df[(df['device_id'] > 2 )&(df['score'] > 0)]
        return df

    def start(self):
        self.row = []
        self.col = []
        self.data = []

        self.map_idx2user = {}
        self.map_idx2item = {}
        self.map_user2idx = {}
        self.map_item2idx = {}
        self.user_cnt = 0
        self.item_cnt = 0

        for data_path in self.data_paths:
            with open(data_path) as f:
                logger.info('start %s'%data_path)
                f.readline()
                cnt = 0
                for line in f:
                    cnt += 1
                    if cnt % 10000 == 0:
                        logger.debug('progress %s %s %s'%(cnt, self.user_cnt, self.item_cnt))
                    self._parse_line(line)

        mat = spMat.coo_matrix((self.data, (self.row, self.col)),
                               shape=(self.user_cnt, self.item_cnt),
                               dtype=numpy.double
                               )
        return RecommenderData(mat,
                               self.map_idx2user,
                               self.map_idx2item,
                               self.map_user2idx,
                               self.map_item2idx)  

    def get_user_idx(self, user_id):
        if user_id in self.map_user2idx:
            return self.map_user2idx[user_id]
        else:
            self.map_user2idx[user_id] = self.user_cnt
            self.map_idx2user[self.user_cnt] = user_id
            self.user_cnt += 1

            return self.user_cnt - 1

    def get_item_idx(self, item_id):
        if item_id in self.map_item2idx:
            return self.map_item2idx[item_id]
        else:
            self.map_item2idx[item_id] = self.item_cnt
            self.map_idx2item[self.item_cnt] = item_id
            self.item_cnt += 1

            return self.item_cnt - 1

    def _parse_line(self, line):
        line = line.strip().split(',')
        dev_id = line[0]
        is_cookie = bool(line[1])
        prefix = ','.join(line[:self.data_prefix_col_num])

        list_ip = ','.join(line[self.data_prefix_col_num:])
        list_ip = list_ip.strip('(){}').split('),(')
        list_ip = [ele.split(',')[0] for ele in list_ip]


        if is_cookie:
            self.row.append(self.get_user_idx(dev_id))
            self.col.append(self.get_item_idx('cookie'))
            self.data.append(1.)

        if len(list_ip) == 0:
            return

        for ip_ele in list_ip:
            if ip_ele in self.pd_active_ip.index:
                self.row.append(self.get_user_idx(dev_id))
                self.col.append(self.get_item_idx(ip_ele))
                self.data.append(self.pd_active_ip.ix[ip_ele, 'score'])
                

def run_svd(recommend_data):

    logger.info('start svd')
    svd = SVD(recommend_data)
    svd.fit(latent_factor=100,
            max_iter=30)

    logger.info('start pickle')
    with open('result.svd', 'wb') as f:
        pickle.dump(svd, f, protocol=pickle.HIGHEST_PROTOCOL)
    return svd

from sklearn.cluster import MiniBatchKMeans

from numba import jit

@jit('void(f8[:, :], i8[:, :], i8[:], f8[: ,:])', nopython=True)
def make_dist_flat(distancess, indicess, label_indices, ret_matrix):
    cnt = 0
    for idx in range(label_indices.shape[0]):
        for i in range(indicess.shape[1]):
            n_idx = indicess[idx, i]
            if idx == n_idx:
                continue
            ret_matrix[cnt, 0] = float(idx)
            ret_matrix[cnt, 1] = float(n_idx)
            ret_matrix[cnt, 2] = distancess[idx, i]
            cnt += 1
            break

class NeaestNode(object):

    def __init__(self,
                 matrix,
                 map_idx2user,
                 category_num=10,
                 category_thresh=1000,
                 nbrs_num=10):
        self.matrix = matrix
        self.category_num = category_num
        self.category_thresh = category_thresh
        self.map_idx2user = map_idx2user
        self.nbrs_num = nbrs_num
        self.label_index = {}
        self.category_index = 0

    def _help_split_categories(self, matrix, list_index):

        label_index = self._kmeans(matrix)
        
        for label, label_indices in label_index.items():
            if len(label_indices) > self.category_thresh:
                if len(list_index) == len(label_indices):
                    logger.debug('cannot split %s'%(len(label_indices)))
                    continue
                sub_matrix = matrix[label_indices]
                sub_index = list_index[label_indices]
                self._help_split_categories(sub_matrix, sub_index)
            else:
                logger.debug('cluster: %s num:%s'%(self.category_index,
                                                  len(label_indices)))
                self.label_index[self.category_index] = \
                    [list_index[i] for i in label_indices]
                self.category_index += 1
                
    def _split_categories(self):
        logger.debug('start')
        self._help_split_categories(self.matrix,
                                    numpy.arange(self.matrix.shape[0]))
        logger.debug('cluster: %s'%(self.category_index))
        logger.debug('end')

    def _kmeans(self, matrix):

        logger.debug('kmeans {} size'.format(matrix.shape[0]))
        labels = MiniBatchKMeans(n_clusters=self.category_num,
                                 batch_size=self.category_num*10,
                                 max_iter=max(20, int(self.category_num / 10 * 2))
                                 ).fit_predict(matrix)
        label_index = {}
        for i, label in enumerate(labels):
            if label in label_index:
                label_index[label].append(i)
            else:
                label_index[label] = [i]

        return label_index

    def fit_predict(self, file_path):
        logger.debug('start')
        self._split_categories()
        out = open(file_path, 'wb')
        out.close()

        manager = mp.Manager()
        queue = manager.Queue()
        pool = mp.Pool()
        watcher = pool.apply_async(write_listener,
                                   (queue, file_path, ))

        for label, label_indices in self.label_index.items():
            logger.debug('num user: %s'%len(label_indices))
            sub_matrix = self.matrix[label_indices]

            if sub_matrix.shape[0] <= self.nbrs_num:
                nbrs_num = sub_matrix.shape[0] - 1
            else:
                nbrs_num = self.nbrs_num


            knn = NearestNeighbors(n_neighbors=nbrs_num + 1,
                                   algorithm='ball_tree', p=2).fit(sub_matrix)

            logger.info('progress label: %s / %s %s'%(label,
                                                       len(self.label_index),
                                                       nbrs_num))
            distancess, indicess = knn.kneighbors(sub_matrix,
                                                  n_neighbors=nbrs_num + 1,
                                                  return_distance=True)
            
            logger.debug('search end')
            data_num = distancess.shape[0]
            ret_matrix = numpy.empty((data_num, 3), dtype='f8')
            flat_data = make_dist_flat(distancess,
                                       indicess, 
                                       numpy.array(label_indices, dtype='i8'),
                                       ret_matrix)
            logger.debug('flat end')
            #jobs = []
            for row in ret_matrix:
                apply(write_worker,(row,
                                    recommend_data.map_idx2user,
                                    queue,))
            """
            cnt = 0
            for idx in xrange(len(label_indices)):
                cnt += 1
                if cnt % 1000 == 0:
                    logger.debug('progress %s %s %s'%(nbrs_num,
                                                      cnt,
                                                      len(label_indices)))

                distances = distancess[idx]
                indices = indicess[idx]
                user_id = self.map_idx2user[label_indices[idx]]

                for i, n_idx in enumerate(indices):
                    if idx == n_idx:
                        continue
                    n_user_id = recommend_data.map_idx2user[label_indices[n_idx]]
                    out.write('%s,%s,%s\n'%(user_id, n_user_id, distances[i]))
            """

        queue.put(QUEUE_STOP_WORD)
        pool.close()
        pool.join()

        logger.debug('end')
        return

if __name__ == '__main__':
    logger.info('load data')
    if 0:
        make_matrix = MakeMatrix(['../data/id_all_ip.csv',
                                  '../data/id_all_property.csv'],
                                 ['../data/score.csv',
                                  '../data/prop_score.csv'],
                                 2)

        recommend_data = make_matrix.start()
        with open('recommend_data.pkl', 'wb') as f:
            pickle.dump(recommend_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('recommend_data.pkl') as f:
            recommend_data = pickle.load(f)
    if 0:
        svd = run_svd(recommend_data)
    else:
        with open('result.svd') as f:
            svd = pickle.load(f)

    logger.info('start lsh')
    knn = NeaestNode(svd.user_matrix,
                     recommend_data.map_idx2user,
                     category_num=100,
                     category_thresh=10000,
                     nbrs_num=1)
    list_nearest = knn.fit_predict('list_nearst.csv')

