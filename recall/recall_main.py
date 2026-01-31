"""
召回主脚本：多路召回（协同过滤、热门、内容相似）
"""
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import faiss

users = pd.read_csv('data/users_feat.csv')
videos = pd.read_csv('data/videos_feat.csv')
inter = pd.read_csv('data/interactions.csv')

# 预计算映射与索引，减少重复计算
video_id_to_author = videos.set_index('video_id')['author_id'].to_dict()
video_id_to_category = videos.set_index('video_id')['category'].to_dict()
video_id_to_tags = videos.set_index('video_id')['tags'].to_dict()

# 预计算类别/标签 -> 视频列表，加速内容召回
category_to_videos = videos.groupby('category')['video_id'].apply(list).to_dict()
tag_to_videos = {}
for vid, tags in video_id_to_tags.items():
    for t in str(tags).split('|'):
        tag_to_videos.setdefault(t, []).append(vid)

# 1. 热门召回：用全站热度兜底
hot_videos = videos.sort_values('video_view_cnt', ascending=False).head(20)['video_id'].tolist()

# 2. 协同过滤（简单UserCF）：基于相似用户行为做召回
user_video = inter[inter['action']=='view'].groupby('user_id')['video_id'].apply(list).to_dict()
def usercf_recall(target_uid, topk=10):
    target_hist = set(user_video.get(target_uid, []))
    sim_cnt = Counter()
    for uid, vids in user_video.items():
        if uid == target_uid: continue
        sim = len(target_hist & set(vids))
        if sim > 0:
            for v in vids:
                if v not in target_hist:
                    sim_cnt[v] += sim
    return [v for v, _ in sim_cnt.most_common(topk)]

# 3. 内容相似召回（同类别/标签）：提高兴趣相关性
def content_recall(target_uid, topk=10):
    # 取用户最近看过的1个视频
    recent = inter[(inter['user_id'] == target_uid) & (inter['action'] == 'view')]
    if recent.empty:
        return []
    # 取最近一次观看
    last_vid = recent.sort_values('timestamp', ascending=False).iloc[0]['video_id']
    cat = video_id_to_category.get(last_vid, None)
    first_tag = str(video_id_to_tags.get(last_vid, '')).split('|')[0]
    sim_vids = []
    if cat in category_to_videos:
        sim_vids.extend(category_to_videos[cat])
    if first_tag in tag_to_videos:
        sim_vids.extend(tag_to_videos[first_tag])
    # 去重并截断
    seen = set()
    res = []
    for v in sim_vids:
        if v not in seen:
            res.append(v)
            seen.add(v)
        if len(res) >= topk:
            break
    return res

# 4. implicit(ALS) + FAISS 向量召回：矩阵分解得到向量，再用ANN检索
def implicit_faiss_recall(topk=10):
    # 构建user_id、video_id到索引的映射
    user_ids = users['user_id'].unique().tolist()
    video_ids = videos['video_id'].unique().tolist()
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    video_id_to_idx = {v: i for i, v in enumerate(video_ids)}

    # 构建交互矩阵（隐式反馈）
    weight_map = {'view': 1.0, 'like': 2.0, 'share': 3.0}
    inter['weight'] = inter['action'].map(weight_map).fillna(1.0)
    row = inter['user_id'].map(user_id_to_idx)
    col = inter['video_id'].map(video_id_to_idx)
    data = inter['weight']
    # implicit 需要 item-user 矩阵（物品在行，用户在列）
    mat = coo_matrix((data, (row, col)), shape=(len(user_ids), len(video_ids))).tocsr()
    item_user = mat.T.tocsr()

    # 训练ALS模型（小数据，CPU可跑）
    model = AlternatingLeastSquares(factors=32, iterations=10, regularization=0.01)
    model.fit(item_user)

    # 提取用户与视频向量
    item_emb = model.item_factors
    user_emb = model.user_factors

    # 构建FAISS索引（内积相似度）
    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    # 归一化向量，内积近似余弦相似度
    faiss.normalize_L2(item_emb)
    index.add(item_emb.astype('float32'))

    # 对每个用户召回topk
    recall = {}
    for uid in user_ids:
        uidx = user_id_to_idx[uid]
        uvec = user_emb[uidx:uidx+1].astype('float32')
        faiss.normalize_L2(uvec)
        scores, idxs = index.search(uvec, topk)
        vids = [video_ids[i] for i in idxs.flatten().tolist()]
        recall[uid] = vids
    return recall

# 5. 双塔模型 + FAISS 召回：深度向量召回
def two_tower_faiss_recall(topk=10, neg_ratio=2):
    # 构建user_id、video_id到索引的映射
    user_ids = users['user_id'].unique().tolist()
    video_ids = videos['video_id'].unique().tolist()
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    video_id_to_idx = {v: i for i, v in enumerate(video_ids)}

    # 正样本：like/share，弱正样本：view
    pos = inter.copy()
    pos['label'] = pos['action'].map({'like': 1.0, 'share': 1.0, 'view': 0.3}).fillna(0.3)
    pos = pos[['user_id', 'video_id', 'label']]

    # 负采样：随机采样用户未互动的视频
    neg_samples = []
    user_hist_map = inter.groupby('user_id')['video_id'].apply(set).to_dict()
    all_videos_set = set(video_ids)
    for uid in user_ids:
        user_hist = user_hist_map.get(uid, set())
        cand = list(all_videos_set - user_hist)
        if not cand:
            continue
        sample_size = min(len(cand), neg_ratio)
        for vid in np.random.choice(cand, size=sample_size, replace=False):
            neg_samples.append({'user_id': uid, 'video_id': vid, 'label': 0.0})
    neg = pd.DataFrame(neg_samples)

    # 训练数据：正负样本拼接
    train_df = pd.concat([pos, neg], ignore_index=True)
    train_df['user_idx'] = train_df['user_id'].map(user_id_to_idx)
    train_df['video_idx'] = train_df['video_id'].map(video_id_to_idx)

    # 双塔模型：学习用户/视频嵌入
    user_input = keras.Input(shape=(), dtype='int32', name='user')
    item_input = keras.Input(shape=(), dtype='int32', name='item')
    emb_dim = 16
    user_emb_layer = keras.layers.Embedding(len(user_ids), emb_dim, name='user_emb')
    item_emb_layer = keras.layers.Embedding(len(video_ids), emb_dim, name='item_emb')
    uvec = user_emb_layer(user_input)
    ivec = item_emb_layer(item_input)
    uvec = keras.layers.Flatten()(uvec)
    ivec = keras.layers.Flatten()(ivec)
    score = keras.layers.Dot(axes=1)([uvec, ivec])
    out = keras.layers.Activation('sigmoid')(score)
    model = keras.Model([user_input, item_input], out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(
        [train_df['user_idx'].values, train_df['video_idx'].values],
        train_df['label'].values,
        epochs=5,
        batch_size=32,
        verbose=0
    )

    # 获取用户和视频向量
    user_emb = user_emb_layer.get_weights()[0]
    item_emb = item_emb_layer.get_weights()[0]

    # FAISS召回：根据用户向量检索视频向量
    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(item_emb)
    index.add(item_emb.astype('float32'))

    recall = {}
    for uid in user_ids:
        uidx = user_id_to_idx[uid]
        uvec = user_emb[uidx:uidx+1].astype('float32')
        faiss.normalize_L2(uvec)
        _, idxs = index.search(uvec, topk)
        vids = [video_ids[i] for i in idxs.flatten().tolist()]
        recall[uid] = vids
    return recall

# 召回主流程：多路召回去重融合
recall_result = {}
implicit_recall = implicit_faiss_recall(topk=5)
two_tower_recall = two_tower_faiss_recall(topk=5)
def merge_dedupe(lists, limit=20):
    seen = set()
    out = []
    for lst in lists:
        for v in lst:
            if v not in seen:
                out.append(v)
                seen.add(v)
            if len(out) >= limit:
                return out
    return out


for uid in users['user_id']:
    recall_result[uid] = merge_dedupe([
        hot_videos[:5],
        usercf_recall(uid, 5),
        content_recall(uid, 5),
        implicit_recall.get(uid, []),
        two_tower_recall.get(uid, [])
    ], limit=20)

# 保存召回结果
recall_df = []
for uid, vids in recall_result.items():
    for vid in vids:
        recall_df.append({'user_id': uid, 'video_id': vid})
pd.DataFrame(recall_df).to_csv('data/recall_result.csv', index=False)
print('召回完成，已保存至data/recall_result.csv')
