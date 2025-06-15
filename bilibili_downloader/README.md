# B站音频下载器

这是一个用于下载B站视频音频的Python工具。它支持下载指定UP主的所有视频音频，并提供音频转文字功能。

## 功能特点

- 支持获取指定UP主的所有视频音频
- 自动选择最高音质的音频流
- 显示下载进度条
- 支持断点续传
- 自动跳过已下载的文件
- 支持将音频文件转换为文字（使用FunASR）

## 安装要求

- Python 3.7+
- FFmpeg（用于音频处理）
- CUDA（可选，用于GPU加速语音识别）
- 依赖包（可通过requirements.txt安装）

## 安装步骤

1. 安装 FFmpeg：
   - Windows: 
     ```bash
     # 使用 chocolatey
     choco install ffmpeg
     
     # 或手动下载安装
     # 从 https://www.gyan.dev/ffmpeg/builds/ 下载
     # 解压后将bin目录添加到系统环境变量
     ```
   - Linux:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - macOS:
     ```bash
     brew install ffmpeg
     ```

2. 克隆或下载本项目到本地

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

4. 安装 FunASR（语音识别）：
```bash
pip install funasr
```

## 使用方法

### 1. 下载音频

```bash
python bilibili_audio_downloader.py
```

运行后，程序会提示输入UP主的uid，然后自动下载该UP主的所有视频音频。

### 2. 音频转文字

```bash
python batch_asr.py
```

这将处理 `BiliAudio` 目录下的所有音频文件，并将识别结果保存在 `BiliAudio/txt` 目录下。

支持的音频格式：
- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)

### 3. 作为模块使用

```python
from bilibili_audio_downloader import BilibiliAudioDownloader

# 初始化下载器（可选：传入SESSDATA以获取更高质量音频）
downloader = BilibiliAudioDownloader()

# 获取UP主的所有视频
uid = "74121740"  # 替换为目标UP主的uid
bvids = downloader.get_user_videos(uid)

# 下载所有视频的音频
for bvid in bvids:
    downloader.download_video_audio(bvid)
```

## 输出文件

- 下载的音频文件将保存在 `BiliAudio` 目录下
- 文件名格式：`{BV号}_{视频标题}_{发布时间}.m4a`
- 转写文本文件将保存在 `BiliAudio/txt` 目录下
- 文本文件名格式：`{BV号}_{视频标题}_{发布时间}.txt`

## 注意事项

1. 建议使用自己的SESSDATA以获取更高质量的音频流
2. 请遵守B站的使用条款和版权规定
3. 下载时请控制频率，避免对服务器造成压力
4. 确保有足够的磁盘空间存储音频文件
5. 使用语音识别功能需要安装FFmpeg
6. 语音识别支持GPU加速，需要安装CUDA
7. 语音识别过程可能需要较长时间，请耐心等待
8. 建议使用GPU进行语音识别以提高速度

## 常见问题

1. 如果遇到下载失败，请检查：
   - 网络连接是否正常
   - UP主的uid是否正确
   - 是否有足够的磁盘空间

2. 如果遇到权限问题，请确保：
   - 有写入目标目录的权限
   - 使用的SESSDATA是否有效

3. 如果遇到FFmpeg相关错误：
   - 确保FFmpeg已正确安装
   - 确保FFmpeg已添加到系统环境变量
   - 尝试重启终端或IDE

4. 如果遇到语音识别问题：
   - 确保已安装FunASR
   - 检查CUDA是否正确安装（如果使用GPU）
   - 检查音频文件格式是否支持
   - 检查音频文件是否可以正常打开和读取
   - 检查是否有足够的系统内存

## 许可证

MIT License 