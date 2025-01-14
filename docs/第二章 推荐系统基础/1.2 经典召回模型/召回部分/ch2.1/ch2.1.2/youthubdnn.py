# import numpy as np
# import random
# from tqdm import tqdm
# import pandas as pd

# def gen_data_set(data, seq_max_len=50, negsample=0):
#     data.sort_values("timestamp", inplace=True)# 按照时间戳排序
#     item_ids = data['movie_id'].unique()# 获取所有item_id(视频id	)
#     item_id_genres_map = dict(zip(data['movie_id'].values, data['genres'].values))# 获取item_id到genre的映射
#     train_set = []
#     test_set = []
#     for reviewerID, hist in tqdm(data.groupby('user_id')):# 按照user_id分组
#         pos_list = hist['movie_id'].tolist()# 获取用户观看过的视频id列表
#         genres_list = hist['genres'].tolist()# 获取用户观看过的视频的genre列表
#         rating_list = hist['rating'].tolist()# 获取用户观看过的视频的评分列表

#         if negsample > 0:
#             candidate_set = list(set(item_ids) - set(pos_list))# 获取所有视频id中用户没有观看过的视频id列表
#             neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)# 从候选集中随机选择negsample个视频id作为负样本
#         for i in range(1, len(pos_list)):
#             hist = pos_list[:i]# 获取用户观看过的视频id列表的前i个视频id
#             genres_hist = genres_list[:i]# 获取用户观看过的视频的genre列表的前i个genre
#             seq_len = min(i, seq_max_len)# 获取用户观看过的视频id列表的长度与seq_max_len的最小值
#             if i != len(pos_list) - 1:# 如果用户观看过的视频id列表的最后一个视频id不是当前视频id
#                 train_set.append((
#                     reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
#                     genres_list[i],
#                     rating_list[i]))# 将用户观看过的视频id列表的前i个视频id，当前视频id，1，用户观看过的视频id列表的前i个视频id的反转，用户观看过的视频id列表的长度，用户观看过的视频的genre列表的前i个genre的反转，当前视频的genre，当前视频的评分添加到训练集中
#                 for negi in range(negsample):
#                     #将用户观看过的视频id列表的前i个视频id，负样本视频id，0，用户观看过的视频id列表的前i个视频id的反转，用户观看过的视频id列表的长度，用户观看过的视频的genre列表的前i个genre的反转，负样本视频的genre添加到训练集中
#                     train_set.append((reviewerID, neg_list[i * negsample + negi], 0, hist[::-1][:seq_len], seq_len,
#                                       genres_hist[::-1][:seq_len], item_id_genres_map[neg_list[i * negsample + negi]]))
#             else:
#                 #将用户观看过的视频id列表的前i个视频id，当前视频id，1，用户观看过的视频id列表的前i个视频id的反转，用户观看过的视频id列表的长度，用户观看过的视频的genre列表的前i个genre的反转，当前视频的genre，当前视频的评分添加到测试集中
#                 test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
#                                  genres_list[i],
#                                  rating_list[i]))

#     random.shuffle(train_set)
#     random.shuffle(test_set)

#     print(len(train_set[0]), len(test_set[0]))

#     return train_set, test_set

# if __name__ == "__main__":
#     data = pd.read_csv("E:/rec_fun/fun-rec/fun-rec-master/docs/ch02/ch2.1/ch2.1.2/sample_data.csv")
#     train_set, test_set = gen_data_set(data)
#     print(train_set)
#     print(test_set)
"""
Author:
    Weichen Shen, weichenswc@163.com
Reference:
Covington P, Adams J, Sargin E. Deep neural networks for youtube recommendations[C]//Proceedings of the 10th ACM conference on recommender systems. 2016: 191-198.
"""
from deepctr.feature_column import build_input_features
from deepctr.layers import DNN
from deepctr.layers.utils import NoMask, combined_dnn_input
from tensorflow.python.keras.models import Model

from ..inputs import input_from_feature_columns, create_embedding_matrix
from ..layers.core import SampledSoftmaxLayer, EmbeddingIndex, PoolingLayer
from ..utils import get_item_embedding, l2_normalize


def YoutubeDNN(user_feature_columns, item_feature_columns,
               user_dnn_hidden_units=(64, 32),
               dnn_activation='relu', dnn_use_bn=False,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, output_activation='linear', temperature=0.05,
               sampler_config=None, seed=1024):
    """Instantiates the YoutubeDNN Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.# 用户特征列
    :param item_feature_columns: An iterable containing item's features used by  the model.#产品特征列
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param output_activation: Activation function to use in output layer
    :param temperature: float. Scaling factor.
    :param sampler_config: negative sample config.
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now YoutubeNN only support 1 item feature like item_id")
    item_feature_name = item_feature_columns[0].name#第一个产品特征列（moive_id）的名字
    item_vocabulary_size = item_feature_columns[0].vocabulary_size#词典大小

    #{'user_id': <tensorflow.python.keras.layers.embeddings.Embedding object at 0x7dbfbf1b83a0>,
    # 'gender': <tensorflow.python.keras.layers.embeddings.Embedding object at 0x7dbfbf1ba7a0>, 
    # 'item_id': <tensorflow.python.keras.layers.embeddings.Embedding object at 0x7dbfbff513c0>}
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed)#创建嵌入矩阵

    user_features = build_input_features(user_feature_columns)#构建输入特征
    #OrderedDict([('user_id', <tf.Tensor 'user_id:0' shape=(None, 1) dtype=int32>), ('gender', <tf.Tensor 'gender:0' shape=(None, 1) dtype=int32>)])
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation=output_activation, seed=seed)(user_dnn_input)
    user_dnn_out = l2_normalize(user_dnn_out)

    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])

    item_embedding_matrix = embedding_matrix_dict[
        item_feature_name]
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))

    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    pooling_item_embedding_weight = l2_normalize(pooling_item_embedding_weight)
    output = SampledSoftmaxLayer(sampler_config._asdict(), temperature)(
        [pooling_item_embedding_weight, user_dnn_out, item_features[item_feature_name]])
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

    return model