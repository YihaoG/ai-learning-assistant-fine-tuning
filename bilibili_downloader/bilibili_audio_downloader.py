#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B站音频下载器
支持从B站视频下载音频，支持批量下载和用户视频列表下载
"""

import os
import sys
import logging
import time
import threading
import multiprocessing
import re
import json
import requests
import pandas as pd
import shutil
import urllib.parse
from functools import reduce
from hashlib import md5

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 混入密钥表
mixinKeyEncTab = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]

# 默认的SESSDATA
SESSDATA = ("0d79b857%2C1764941973%2C0486f%2A62CjAZD1DtaTpeHR3w--9fVGliTuKjp25255Q"
            "IgmGydTORLbRgV2s6oIDhjB2JbUwKF60SVkJhbGFyM0pQc1JxX0ZaYWdCTTZsU2xuQlEta"
            "UFqc3RaTlNiTEVQZWFoRFdCamFjN0x6Rm9NV3pINUs4RzEyQXlod00xdG41cHdFeW85djZuaVlGN1FRIIEC")

class BilibiliAudioDownloader:
    def __init__(self, sessdata=SESSDATA):
        """
        初始化下载器
        :param sessdata: B站登录后的SESSDATA，用于获取更高质量的音频流
        """
        self.sessdata = sessdata
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Referer': 'https://www.bilibili.com/',
            'Cookie': f'SESSDATA={sessdata}' if sessdata else '',
        }

    def get_wbi_keys(self):
        """获取最新的 img_key 和 sub_key"""
        resp = requests.get('https://api.bilibili.com/x/web-interface/nav', headers=self.headers)
        resp.raise_for_status()
        json_content = resp.json()
        img_url: str = json_content['data']['wbi_img']['img_url']
        sub_url: str = json_content['data']['wbi_img']['sub_url']
        img_key = img_url.rsplit('/', 1)[1].split('.')[0]
        sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
        return img_key, sub_key

    def get_mixin_key(self, orig: str):
        """对 imgKey 和 subKey 进行字符顺序打乱编码"""
        return reduce(lambda s, i: s + orig[i], mixinKeyEncTab, '')[:32]

    def enc_wbi(self, params: dict, img_key: str, sub_key: str):
        """为请求参数进行 wbi 签名"""
        mixin_key = self.get_mixin_key(img_key + sub_key)
        curr_time = round(time.time())
        params['wts'] = curr_time
        params = dict(sorted(params.items()))
        params = {
            k: ''.join(filter(lambda chr: chr not in "!'()*", str(v)))
            for k, v in params.items()
        }
        query = urllib.parse.urlencode(params)
        wbi_sign = md5((query + mixin_key).encode()).hexdigest()
        params['w_rid'] = wbi_sign
        return params

    def get_video_info(self, bvid):
        """获取视频的cid和标题"""
        img_key, sub_key = self.get_wbi_keys()
        signed_params = self.enc_wbi(
            params={'bvid': bvid},
            img_key=img_key,
            sub_key=sub_key
        )
        query = urllib.parse.urlencode(signed_params)
        url = f"https://api.bilibili.com/x/web-interface/view?{query}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if data['code'] == 0:
                title = data['data']['title']
                safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()
                cid = data['data']['cid']
                pubtime_arr = time.localtime(data['data']['pubdate'])
                pubtime_str = time.strftime("%Y-%m-%d-%H-%M-%S", pubtime_arr)
                return cid, safe_title, pubtime_str
            else:
                logger.error(f"获取视频信息失败: {data['message']}")
                return None, None, None
        except requests.exceptions.RequestException as e:
            logger.error(f"请求视频信息时发生错误: {e}")
            return None, None, None

    def get_audio_url(self, cid, bvid):
        """获取音频的下载链接"""
        img_key, sub_key = self.get_wbi_keys()
        signed_params = self.enc_wbi(
            params={
                'bvid': bvid,
                'cid': cid,
                'fnval': 80
            },
            img_key=img_key,
            sub_key=sub_key
        )
        query = urllib.parse.urlencode(signed_params)
        url = f"https://api.bilibili.com/x/player/playurl?{query}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if data['code'] == 0:
                audios = data['data']['dash']['audio']
                best_audio = max(audios, key=lambda x: x['bandwidth'])
                logger.info(f"已选择最高音质码率: {best_audio['bandwidth']} bps")
                return best_audio['baseUrl']
            else:
                logger.error(f"获取播放链接失败: {data['message']}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"请求播放链接时发生错误: {e}")
            return None

    def download_file(self, url, filename):
        """下载文件并显示进度"""
        temp_filename = filename + ".tmp"
        try:
            response = requests.get(url, headers=self.headers, stream=True, timeout=20)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            if os.path.exists(filename) and os.path.getsize(filename) == total_size:
                logger.info(f"文件 '{filename}' 已存在且完整，跳过下载。")
                return True

            logger.info(f"开始下载: {filename} (大小: {total_size / (1024 * 1024):.2f}MB)")
            with open(temp_filename, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = int(50 * downloaded_size / total_size) if total_size else 0
                        print(
                            f"\r[{'=' * progress}{' ' * (50 - progress)}] {downloaded_size / (1024 * 1024):.2f}MB / {total_size / (1024 * 1024):.2f}MB",
                            end='')
            print()
            shutil.move(temp_filename, filename)
            logger.info(f"下载完成: {filename}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"\n下载 {filename} 时发生错误: {e}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return False

    def download_video_audio(self, bvid, output_dir='BiliAudio'):
        """下载单个视频的音频"""
        logger.info(f"--- 开始处理视频: {bvid} ---")
        cid, title, pubtime = self.get_video_info(bvid)
        if not cid:
            logger.error(f"--- 视频 {bvid} 处理失败 ---\n")
            return False

        logger.info(f"视频标题: {title}")

        audio_url = self.get_audio_url(cid, bvid)
        if not audio_url:
            logger.error(f"--- 视频 {bvid} 处理失败 ---\n")
            return False

        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{bvid}_{title}_{pubtime}.m4a")

        if not self.download_file(audio_url, output_filename):
            logger.error(f"--- 视频 {bvid} 处理失败 ---\n")
            return False

        logger.info(f"音频已成功保存到: {output_filename}")
        logger.info(f"--- 视频 {bvid} 处理完成 ---\n")
        return True

    def get_user_videos(self, uid):
        """
        获取 B 站指定用户的全部视频 BV 号

        参数:
            uid (str): 用户的 mid（用户 ID）

        返回:
            list: 包含所有视频 BV 号的列表
        """
        bvids = []
        page = 1
        pagesize = 30  # 每页最多 30 个视频

        while True:
            img_key, sub_key = self.get_wbi_keys()
            signed_params = self.enc_wbi(
                params={
                    'mid': uid,
                    'pn': page
                },
                img_key=img_key,
                sub_key=sub_key
            )
            query = urllib.parse.urlencode(signed_params)

            # 构造 API 请求 URL
            url = f"https://api.bilibili.com/x/space/wbi/arc/search?{query}"

            # 控制请求频率，避免被封禁
            time.sleep(5)

            try:
                # 发送请求
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()  # 检查请求是否成功

                # 解析 JSON 数据
                data = response.json()
                if data['code'] != 0:
                    print(f"API 请求失败: {data['message']}")
                    break

                # 提取视频列表
                videos = data['data']['list']['vlist']
                if not videos:
                    break  # 没有更多视频了

                # 提取 BV 号
                for video in videos:
                    bvids.append(video['bvid'])

                print(f"已获取第 {page} 页，共 {len(videos)} 个视频")

                # 翻页
                page += 1

                # 控制请求频率，避免被封禁
                time.sleep(10)

            except requests.RequestException as e:
                print(f"请求出错: {e}")
                break
            except (KeyError, json.JSONDecodeError) as e:
                print(f"解析数据出错: {e}")
                break

        return bvids

def main():
    """主函数"""
    # 创建输出目录
    output_dir = 'BiliAudio'
    os.makedirs(output_dir, exist_ok=True)
    # 初始化下载器
    downloader = BilibiliAudioDownloader()

    # 获取用户输入
    uid = input("请输入UP主的uid（例如：74121740）：").strip()
    
    if not uid:
        logger.error("未输入uid，程序退出")
        return

    # 获取用户所有视频的BV号
    logger.info(f"开始获取用户 {uid} 的所有视频...")
    bvids = downloader.get_user_videos(uid)
    
    if not bvids:
        logger.error("未获取到任何视频，程序退出")
        return

    logger.info(f"共获取到 {len(bvids)} 个视频")
    
    # 下载所有视频的音频
    for i, bvid in enumerate(bvids, 1):
        logger.info(f"正在处理第 {i}/{len(bvids)} 个视频")
        downloader.download_video_audio(bvid, output_dir)
        # 添加延时，避免请求过于频繁
        time.sleep(5)

    logger.info("所有视频处理完成！")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 