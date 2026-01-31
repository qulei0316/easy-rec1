"""
数据生成脚本：生成用户、视频、行为日志样例数据
可直接在本地Mac运行（支持参数配置规模）
"""
import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


def parse_args():
    parser = argparse.ArgumentParser(description='生成推荐系统样例数据')
    parser.add_argument('--users', type=int, default=200, help='用户数量')
    parser.add_argument('--videos', type=int, default=2000, help='视频数量')
    parser.add_argument('--interactions', type=int, default=20000, help='交互数量')
    return parser.parse_args()


def main():
    args = parse_args()

    # 用户数据：用于模拟基础用户画像
    user_num = args.users
    users = []
    for i in range(1, user_num + 1):
        users.append({
            'user_id': i,
            'age': random.randint(18, 50),
            'gender': random.choice(['M', 'F']),
            'register_days': random.randint(1, 1000)
        })
    pd.DataFrame(users).to_csv('data/users.csv', index=False)

    # 视频数据：模拟内容库与作者分布
    video_num = args.videos
    categories = ['美食', '旅游', '音乐', '运动', '科技']
    tags_pool = ['搞笑', '治愈', '高能', '热门', '冷门', '新手', '推荐']
    videos = []
    for i in range(1, video_num + 1):
        videos.append({
            'video_id': i,
            'author_id': random.randint(1, max(10, video_num // 20)),
            'category': random.choice(categories),
            'publish_days': random.randint(1, 365),
            'tags': '|'.join(random.sample(tags_pool, k=2))
        })
    pd.DataFrame(videos).to_csv('data/videos.csv', index=False)

    # 行为日志：模拟用户对视频的交互
    interactions = []
    actions = ['view', 'like', 'share']
    for _ in range(args.interactions):
        u = random.randint(1, user_num)
        v = random.randint(1, video_num)
        action = random.choices(actions, weights=[0.7, 0.2, 0.1])[0]
        ts = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        interactions.append({
            'user_id': u,
            'video_id': v,
            'action': action,
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S')
        })
    pd.DataFrame(interactions).to_csv('data/interactions.csv', index=False)

    print('数据生成完毕，已保存至data/目录')


if __name__ == '__main__':
    main()
