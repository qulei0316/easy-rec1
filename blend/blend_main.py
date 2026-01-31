"""
策略融合主脚本：多路融合、打散（企业级风格简化版）
"""
import pandas as pd
import random

random.seed(42)

rerank = pd.read_csv('data/rerank_result.csv')
videos = pd.read_csv('data/videos.csv')
users_feat = pd.read_csv('data/users_feat.csv')

# 尝试读取精排分数用于排序（若不存在则用rerank顺序）
try:
    rank_df = pd.read_csv('data/rank_result.csv')
    rank_score_map = {(r.user_id, r.video_id): r.rank_score for r in rank_df.itertuples()}
except Exception:
    rank_score_map = {}

# 热门与新鲜池
videos_feat = None
try:
    videos_feat = pd.read_csv('data/videos_feat.csv')
except Exception:
    videos_feat = videos

hot_pool = videos_feat.sort_values('video_view_cnt', ascending=False)['video_id'].tolist()
fresh_pool = videos.sort_values('publish_days', ascending=True)['video_id'].tolist()

video_meta = videos.set_index('video_id')[['author_id', 'category', 'publish_days']].to_dict('index')
user_activity = users_feat.set_index('user_id')[['user_view_cnt', 'user_like_cnt', 'user_share_cnt']].sum(axis=1).to_dict()


def pick_from_pool(pool, seen, author_cnt, cat_cnt, max_author=2, max_cat=4):
    """从候选池中挑一个满足约束的未见视频"""
    for vid in pool:
        if vid in seen:
            continue
        meta = video_meta.get(vid, {})
        author = meta.get('author_id', -1)
        cat = meta.get('category', 'unknown')
        if author_cnt.get(author, 0) >= max_author:
            continue
        if cat_cnt.get(cat, 0) >= max_cat:
            continue
        return vid
    return None


def blend_for_user(uid, base_list, max_len=10):
    # 根据活跃度调整策略配比
    activity = user_activity.get(uid, 0)
    if activity < 5:
        quotas = {'base': 6, 'hot': 3, 'fresh': 1}
    elif activity < 20:
        quotas = {'base': 7, 'hot': 2, 'fresh': 1}
    else:
        quotas = {'base': 7, 'hot': 1, 'fresh': 2}

    # 基于精排分数排序（若有）
    if rank_score_map:
        base_list = sorted(base_list, key=lambda v: rank_score_map.get((uid, v), 0), reverse=True)

    seen = set()
    author_cnt = {}
    cat_cnt = {}
    result = []

    base_pool = list(base_list)
    hot_pool_user = list(hot_pool)
    fresh_pool_user = list(fresh_pool)

    # 轮询各池进行融合
    while len(result) < max_len and sum(quotas.values()) > 0:
        for pool_name in ['base', 'hot', 'fresh']:
            if quotas[pool_name] <= 0 or len(result) >= max_len:
                continue
            if pool_name == 'base':
                vid = pick_from_pool(base_pool, seen, author_cnt, cat_cnt)
            elif pool_name == 'hot':
                vid = pick_from_pool(hot_pool_user, seen, author_cnt, cat_cnt)
            else:
                vid = pick_from_pool(fresh_pool_user, seen, author_cnt, cat_cnt)

            if vid is None:
                quotas[pool_name] = 0
                continue

            seen.add(vid)
            meta = video_meta.get(vid, {})
            author = meta.get('author_id', -1)
            cat = meta.get('category', 'unknown')
            author_cnt[author] = author_cnt.get(author, 0) + 1
            cat_cnt[cat] = cat_cnt.get(cat, 0) + 1
            result.append(vid)
            quotas[pool_name] -= 1

    # 兜底：不足时用base_pool填充
    if len(result) < max_len:
        for vid in base_pool:
            if vid in seen:
                continue
            result.append(vid)
            if len(result) >= max_len:
                break
    return result


blend_result = []
for uid, group in rerank.groupby('user_id'):
    base_vids = group['video_id'].tolist()
    final_vids = blend_for_user(uid, base_vids, max_len=10)
    for v in final_vids:
        blend_result.append({'user_id': uid, 'video_id': v})

pd.DataFrame(blend_result).to_csv('data/blend_result.csv', index=False)
print('策略融合完成，已保存至data/blend_result.csv')
