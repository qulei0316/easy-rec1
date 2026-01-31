"""
精排主脚本：DNN模型（Tensorflow），多目标打分（旧版保留）
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 读取粗排结果和特征
prerank = pd.read_csv('data/prerank_result.csv')
users = pd.read_csv('data/users_feat.csv')
videos = pd.read_csv('data/videos_feat.csv')
inter = pd.read_csv('data/interactions.csv')

# 构造训练样本（正样本：like，负样本：无like）
prerank['label'] = 0
inter_pos = inter[inter['action']=='like'][['user_id','video_id']]
inter_pos['label'] = 1
train_df = pd.concat([prerank, inter_pos], ignore_index=True).drop_duplicates(['user_id','video_id'], keep='last')
train_df = train_df.merge(users, on='user_id', how='left').merge(videos, on='video_id', how='left').fillna(0)

feat_cols = ['age','register_days','is_male','user_view_cnt','user_like_cnt','user_share_cnt',
             'publish_days','tag_count','video_view_cnt','video_like_cnt','video_share_cnt']
X = train_df[feat_cols].values
y = train_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 简单DNN：多层感知机做点击/喜欢预测
model = keras.Sequential([
    keras.layers.Input(shape=(len(feat_cols),)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=2, validation_data=(X_test, y_test))

# 精排打分：对粗排候选打分
prerank = prerank.merge(users, on='user_id', how='left').merge(videos, on='video_id', how='left').fillna(0)
X_pred = prerank[feat_cols].values
prerank['rank_score'] = model.predict(X_pred, batch_size=32).flatten()

# 每用户取TopN：保留15条进入重排
rank_result = prerank.sort_values(['user_id','rank_score'], ascending=[True,False]).groupby('user_id').head(15)
rank_result[['user_id','video_id','rank_score']].to_csv('data/rank_result.csv', index=False)
print('精排完成，已保存至data/rank_result.csv')
