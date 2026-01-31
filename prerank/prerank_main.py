"""
粗排主脚本：LR模型，快速过滤候选
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 读取召回结果和特征
recall = pd.read_csv('data/recall_result.csv')
users = pd.read_csv('data/users_feat.csv')
videos = pd.read_csv('data/videos_feat.csv')
inter = pd.read_csv('data/interactions.csv')

# 构造训练样本（正样本：like；负样本：曝光未互动 + 负采样补充）
pos = inter[inter['action'] == 'like'][['user_id', 'video_id']].copy()
pos['label'] = 1

# 曝光未互动（这里用 view 作为曝光，未点赞视作负）
exposure = inter[inter['action'] == 'view'][['user_id', 'video_id']].copy()
exposure = exposure.merge(pos[['user_id', 'video_id']], on=['user_id', 'video_id'], how='left', indicator=True)
exposure = exposure[exposure['_merge'] == 'left_only'][['user_id', 'video_id']]
exposure['label'] = 0

# 负采样：补充“未曝光”负样本
all_videos = videos['video_id'].unique().tolist()
all_videos_set = set(all_videos)
user_hist = inter.groupby('user_id')['video_id'].apply(set).to_dict()

neg_samples = []
neg_ratio = 3  # 每个用户采样的负样本数量（可调）
for uid in users['user_id']:
    seen = user_hist.get(uid, set())
    cand = list(all_videos_set - seen)
    if not cand:
        continue
    sample_size = min(len(cand), neg_ratio)
    for vid in np.random.choice(cand, size=sample_size, replace=False):
        neg_samples.append({'user_id': uid, 'video_id': vid, 'label': 0})

neg = pd.DataFrame(neg_samples)
train_df = pd.concat([pos, exposure, neg], ignore_index=True).drop_duplicates(['user_id', 'video_id'], keep='last')

# 合并特征：用户与视频侧特征拼接
train_df = train_df.merge(users, on='user_id', how='left').merge(videos, on='video_id', how='left')

# 简单特征：数值型统计特征
feat_cols = ['age','register_days','is_male','user_view_cnt','user_like_cnt','user_share_cnt',
             'publish_days','tag_count','video_view_cnt','video_like_cnt','video_share_cnt']
train_df = train_df.fillna(0)

# 训练LR：轻量粗排模型
X = train_df[feat_cols]
y = train_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
print(f'粗排LR模型AUC: {auc:.4f}')

# 粗排打分：为召回候选打分
recall = recall.merge(users, on='user_id', how='left').merge(videos, on='video_id', how='left').fillna(0)
recall['prerank_score'] = lr.predict_proba(recall[feat_cols])[:,1]

# 每用户取TopN：保留30条进入精排
prerank_result = recall.sort_values(['user_id','prerank_score'], ascending=[True,False]).groupby('user_id').head(30)
prerank_result[['user_id','video_id','prerank_score']].to_csv('data/prerank_result.csv', index=False)
print('粗排完成，已保存至data/prerank_result.csv')
