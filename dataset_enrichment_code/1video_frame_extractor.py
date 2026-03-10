#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP4视频帧提取脚本
按照每秒5帧的频率提取视频帧，并保存为1280×720分辨率的图片
"""

import os
import cv2
import glob
import re
from pathlib import Path

def extract_number_from_filename(filename):
    """
    从文件名中提取首个数字
    
    Args:
        filename (str): 文件名
        
    Returns:
        int: 提取的数字，如果没找到则返回0
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def resize_frame(frame, target_width=1280, target_height=720):
    """
    将帧调整为指定分辨率
    
    Args:
        frame: 输入帧
        target_width (int): 目标宽度，默认1280
        target_height (int): 目标高度，默认720
        
    Returns:
        resized_frame: 调整后的帧
    """
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def extract_frames_from_videos(input_folder, output_folder, fps=5, target_width=1280, target_height=720):
    """
    从指定文件夹中的所有MP4文件提取帧并调整分辨率
    
    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
        fps (int): 每秒提取的帧数，默认5帧
        target_width (int): 目标宽度，默认1280
        target_height (int): 目标高度，默认720
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有MP4文件
    mp4_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    mp4_files.sort()  # 按文件名排序
    
    if not mp4_files:
        print(f"在文件夹 {input_folder} 中没有找到MP4文件")
        return
    
    print(f"找到 {len(mp4_files)} 个MP4文件")
    
    for video_path in mp4_files:
        video_name = os.path.basename(video_path)
        print(f"\n正在处理: {video_name}")
        
        # 从文件名中提取首个数字作为图片命名前缀
        video_number = extract_number_from_filename(video_name)
        print(f"  提取的视频编号: {video_number}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            continue
        
        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  原始帧率: {original_fps:.2f} FPS")
        print(f"  原始分辨率: {original_width}×{original_height}")
        print(f"  目标分辨率: {target_width}×{target_height}")
        print(f"  总帧数: {total_frames}")
        print(f"  视频时长: {duration:.2f} 秒")
        
        # 计算帧间隔
        frame_interval = int(original_fps / fps)
        if frame_interval < 1:
            frame_interval = 1
        
        print(f"  每 {frame_interval} 帧提取一帧")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔提取帧
            if frame_count % frame_interval == 0:
                saved_count += 1
                
                # 调整帧的分辨率
                resized_frame = resize_frame(frame, target_width, target_height)
                
                # 构造文件名: 视频编号-帧编号 (使用从文件名提取的数字)
                filename = f"{video_number}-{saved_count}.jpg"
                output_path = os.path.join(output_folder, filename)
                
                # 保存调整后的图片
                cv2.imwrite(output_path, resized_frame)
                
                if saved_count % 50 == 0:  # 每50帧显示一次进度
                    print(f"  已保存 {saved_count} 帧...")
            
            frame_count += 1
        
        cap.release()
        print(f"  完成! 共保存了 {saved_count} 帧图片")
    
    print(f"\n所有视频处理完成! 图片已保存到: {output_folder}")
    print(f"所有图片分辨率已调整为: {target_width}×{target_height}")

def main():
    """主函数"""
    # 配置路径
    # input_folder = "/data/LLM/WSC/dataset_video/data_video"  # 输入文件夹路径
    # output_folder = "/data/LLM/WSC/dataset_video/extracted_frames"  # 输出文件夹路径

    input_folder = "/data/LLM/WSC/dataset_video/data_video_test"  # 输入文件夹路径
    output_folder = "/data/LLM/WSC/dataset_video/extracted_frames_test"  # 输出文件夹路径
    
    # 分辨率配置
    target_width = 1280
    target_height = 720
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"输入文件夹不存在: {input_folder}")
        print("请修改 input_folder 变量为正确的路径")
        return
    
    print("=" * 50)
    print("MP4视频帧提取工具")
    print("=" * 50)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"提取频率: 每秒5帧")
    print(f"输出分辨率: {target_width}×{target_height}")
    print("=" * 50)
    
    # 开始提取
    extract_frames_from_videos(input_folder, output_folder, fps=5, 
                              target_width=target_width, target_height=target_height)

if __name__ == "__main__":
    main()