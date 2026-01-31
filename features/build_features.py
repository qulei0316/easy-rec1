"""
特征工程脚本：构建用户、视频、交互特征
"""
import pandas as pd
import numpy as np

# 读取数据：用户、视频、交互日志
users = pd.read_csv('data/users.csv')
videos = pd.read_csv('data/videos.csv')
inter = pd.read_csv('data/interactions.csv')

# 用户特征：性别二值化
users['is_male'] = (users['gender'] == 'M').astype(int)

# 视频特征：标签数量
videos['tag_count'] = videos['tags'].apply(lambda x: len(str(x).split('|')))

# 用户-视频交互特征：行为类型 one-hot
inter['view'] = (inter['action'] == 'view').astype(int)
inter['like'] = (inter['action'] == 'like').astype(int)
inter['share'] = (inter['action'] == 'share').astype(int)

# 用户历史行为统计：累积观看/点赞/分享次数
user_hist = inter.groupby('user_id').agg({'view':'sum','like':'sum','share':'sum'}).reset_index()
user_hist.columns = ['user_id','user_view_cnt','user_like_cnt','user_share_cnt']
users = users.merge(user_hist, on='user_id', how='left').fillna(0)

# 视频历史行为统计：累积观看/点赞/分享次数
video_hist = inter.groupby('video_id').agg({'view':'sum','like':'sum','share':'sum'}).reset_index()
video_hist.columns = ['video_id','video_view_cnt','video_like_cnt','video_share_cnt']
videos = videos.merge(video_hist, on='video_id', how='left').fillna(0)

# 保存特征文件，供召回/排序使用
users.to_csv('data/users_feat.csv', index=False)
videos.to_csv('data/videos_feat.csv', index=False)
print('特征工程完成，已保存至data/目录')
