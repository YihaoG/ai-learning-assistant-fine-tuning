import os
from funasr import AutoModel
import glob
import pathlib

def process_audio_files(input_dir, output_dir):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 FunASR 模型
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc",
        punc_model_revision="v2.0.4",
        device="cuda:0",
    )
    
    # 支持的音频格式
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    
    # 使用 pathlib 处理路径
    input_path = pathlib.Path(input_dir).resolve()
    output_path = pathlib.Path(output_dir).resolve()
    
    print(f"正在扫描目录: {input_path}")
    audio_files = []
    
    # 遍历目录
    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)
    
    print(f"找到 {len(audio_files)} 个音频文件")
    print("文件列表：")
    for file in audio_files:
        print(f"- {file}")
    
    # 处理每个音频文件
    for audio_file in audio_files:
        try:
            print(f"\n正在处理: {audio_file}")
            
            # 检查文件是否存在
            if not audio_file.exists():
                print(f"文件不存在: {audio_file}")
                continue
                
            # 生成输出文件路径
            output_file = output_path / f"{audio_file.stem}.txt"
            
            print(f"开始识别...")
            # 进行语音识别
            # 尝试使用不同的路径格式
            audio_path = str(audio_file.absolute()).replace('\\', '/')
            print(f"使用文件路径: {audio_path}")
            
            # 确保文件可读
            with open(audio_file, 'rb') as f:
                print("文件可以正常打开和读取")
            
            result = model.generate(input=audio_path)
            
            # 保存识别结果
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result[0]['text'])
            
            print(f"已保存结果到: {output_file}")
            
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错: {str(e)}")
            print(f"文件路径: {audio_file.absolute()}")
            # 尝试获取更多文件信息
            try:
                print(f"文件大小: {audio_file.stat().st_size} bytes")
                print(f"文件权限: {oct(audio_file.stat().st_mode)[-3:]}")
                # 尝试读取文件的前几个字节
                with open(audio_file, 'rb') as f:
                    print(f"文件头部字节: {f.read(16)}")
            except Exception as e2:
                print(f"无法获取文件信息: {str(e2)}")

if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = r"E:\audio\BiliAudio"  # 使用原始字符串
    output_directory = r"E:\audio\BiliAudio\txt"
    
    # 确保输入目录存在
    if not os.path.exists(input_directory):
        print(f"输入目录不存在: {input_directory}")
        exit(1)
        
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")
    
    process_audio_files(input_directory, output_directory) 