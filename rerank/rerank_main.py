"""
重排主脚本：多样性、去重、业务规则
"""
import pandas as pd

rank = pd.read_csv('data/rank_result.csv')
videos = pd.read_csv('data/videos.csv')

# 业务规则：每用户最多同一作者2个视频，类别多样性
rerank_result = []
for uid, group in rank.groupby('user_id'):
    vids = []
    author_cnt = {}
    cat_set = set()
    for _, row in group.iterrows():
        vid = row['video_id']
        author = videos[videos['video_id']==vid]['author_id'].values[0]
        cat = videos[videos['video_id']==vid]['category'].values[0]
        # 作者频控：避免同作者过多
        if author_cnt.get(author,0) >= 2:
            continue
        # 类别多样性：避免类别过于集中
        if cat in cat_set and len(cat_set)<5:
            continue
        vids.append(vid)
        author_cnt[author] = author_cnt.get(author,0)+1
        cat_set.add(cat)
        if len(vids) >= 10:
            break
    for v in vids:
        rerank_result.append({'user_id': uid, 'video_id': v})

pd.DataFrame(rerank_result).to_csv('data/rerank_result.csv', index=False)
print('重排完成，已保存至data/rerank_result.csv')
