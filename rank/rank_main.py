"""
精排主脚本：Transformer 精排（序列建模）
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

SEQ_LEN = 15

# 构造训练样本（正样本：like，负样本：无like）
prerank['label'] = 0
inter_pos = inter[inter['action'] == 'like'][['user_id', 'video_id']]
inter_pos['label'] = 1
train_df = pd.concat([prerank, inter_pos], ignore_index=True).drop_duplicates(['user_id', 'video_id'], keep='last')
train_df = train_df.merge(users, on='user_id', how='left').merge(videos, on='video_id', how='left').fillna(0)

# 类别特征映射
train_df['category_id'], cat_uniques = pd.factorize(train_df['category'])

# 构建用户序列（按时间排序）
inter_sorted = inter.sort_values('timestamp')
user_seq_map = inter_sorted.groupby('user_id')['video_id'].apply(list).to_dict()

# video_id 映射（0 用作 padding）
video_ids = videos['video_id'].unique().tolist()
video_id_to_idx = {v: i + 1 for i, v in enumerate(video_ids)}

def build_seq(uid, max_len=SEQ_LEN):
    seq = user_seq_map.get(uid, [])
    seq = [video_id_to_idx.get(v, 0) for v in seq][-max_len:]
    if len(seq) < max_len:
        seq = [0] * (max_len - len(seq)) + seq
    return seq

train_df['seq_ids'] = train_df['user_id'].apply(build_seq)

# 数值特征
num_cols = [
    'age', 'register_days', 'is_male', 'user_view_cnt', 'user_like_cnt', 'user_share_cnt',
    'publish_days', 'tag_count', 'video_view_cnt', 'video_like_cnt', 'video_share_cnt'
]

X = {
    'seq_ids': np.stack(train_df['seq_ids'].values),
    'video_id': train_df['video_id'].map(video_id_to_idx).values,
    'author_id': train_df['author_id'].values,
    'category_id': train_df['category_id'].values,
    'num_feat': train_df[num_cols].values.astype('float32')
}
y = train_df['label'].values

X_train_idx, X_test_idx = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
X_train = {k: v[X_train_idx] for k, v in X.items()}
X_test = {k: v[X_test_idx] for k, v in X.items()}
y_train, y_test = y[X_train_idx], y[X_test_idx]

def build_transformer_ranker(video_vocab, author_num, category_num, num_dim, seq_len):
    # Transformer 关键超参
    emb_dim = 32       # embedding 维度
    num_heads = 4      # 多头注意力头数
    ff_dim = 64        # 前馈网络隐藏层维度

    # 输入：用户最近行为序列 + 当前候选视频/作者/类目 + 数值特征
    seq_in = keras.Input(shape=(seq_len,), dtype='int32', name='seq_ids')
    video_in = keras.Input(shape=(), dtype='int32', name='video_id')
    author_in = keras.Input(shape=(), dtype='int32', name='author_id')
    cat_in = keras.Input(shape=(), dtype='int32', name='category_id')
    num_in = keras.Input(shape=(num_dim,), dtype='float32', name='num_feat')

    # 1) 序列 embedding（把视频ID序列映射成向量序列）
    # mask_zero=True 表示0是padding，会被注意力忽略
    item_emb_layer = keras.layers.Embedding(video_vocab + 1, emb_dim, mask_zero=True)
    # 位置编码（让模型知道序列顺序）
    pos_emb_layer = keras.layers.Embedding(seq_len, emb_dim)

    seq_emb = item_emb_layer(seq_in)
    pos_ids = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = pos_emb_layer(pos_ids)
    seq_emb = seq_emb + pos_emb

    # 2) 多头自注意力：学习用户最近行为之间的关联
    attn_out = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)(seq_emb, seq_emb)
    attn_out = keras.layers.Add()([seq_emb, attn_out])
    attn_out = keras.layers.LayerNormalization()(attn_out)

    # 3) 前馈网络：非线性变换
    ff = keras.layers.Dense(ff_dim, activation='relu')(attn_out)
    ff = keras.layers.Dense(emb_dim)(ff)
    ff = keras.layers.Add()([attn_out, ff])
    ff = keras.layers.LayerNormalization()(ff)

    # 4) 池化成固定长度的用户序列向量
    seq_vec = keras.layers.GlobalAveragePooling1D()(ff)

    # 5) 当前候选视频侧特征 embedding
    video_emb = item_emb_layer(video_in)
    video_emb = keras.layers.Flatten()(video_emb)
    author_emb = keras.layers.Embedding(author_num + 1, emb_dim)(author_in)
    author_emb = keras.layers.Flatten()(author_emb)
    cat_emb = keras.layers.Embedding(category_num + 1, emb_dim)(cat_in)
    cat_emb = keras.layers.Flatten()(cat_emb)

    # 6) 拼接：用户序列向量 + 视频向量 + 作者/类目 + 数值特征
    x = keras.layers.Concatenate()([seq_vec, video_emb, author_emb, cat_emb, num_in])
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)

    # 7) 输出点击/喜欢概率
    model = keras.Model([seq_in, video_in, author_in, cat_in, num_in], out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


model = build_transformer_ranker(
    video_vocab=len(video_ids),
    author_num=videos['author_id'].max(),
    category_num=len(cat_uniques),
    num_dim=len(num_cols),
    seq_len=SEQ_LEN
)

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2, validation_data=(X_test, y_test))

prerank = prerank.merge(users, on='user_id', how='left').merge(videos, on='video_id', how='left').fillna(0)
prerank['category_id'] = prerank['category'].map({v: i for i, v in enumerate(cat_uniques)}).fillna(0).astype(int)
prerank['seq_ids'] = prerank['user_id'].apply(build_seq)
X_pred = {
    'seq_ids': np.stack(prerank['seq_ids'].values),
    'video_id': prerank['video_id'].map(video_id_to_idx).fillna(0).astype(int).values,
    'author_id': prerank['author_id'].values,
    'category_id': prerank['category_id'].values,
    'num_feat': prerank[num_cols].values.astype('float32')
}
prerank['rank_score'] = model.predict(X_pred, batch_size=64).flatten()

# 每用户取TopN
rank_result = prerank.sort_values(['user_id', 'rank_score'], ascending=[True, False]).groupby('user_id').head(15)
rank_result[['user_id', 'video_id', 'rank_score']].to_csv('data/rank_result.csv', index=False)
print('精排完成，已保存至data/rank_result.csv')
