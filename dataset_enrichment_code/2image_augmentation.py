#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import glob
from pathlib import Path
import argparse

def apply_random_augmentations(image_path, output_path, 
                             brightness_range=(0.8, 1.2),
                             contrast_range=(0.85, 1.15),
                             noise_range=(0.01, 0.03),
                             blur_range=(0, 1.5)):
    """
    对单张图片应用随机增强
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        brightness_range: 亮度调整范围 (最小值, 最大值)
        contrast_range: 对比度调整范围 (最小值, 最大值)
        noise_range: 噪声强度范围 (最小值, 最大值)
        blur_range: 模糊半径范围 (最小值, 最大值)
    """
    try:
        # 打开图片
        image = Image.open(image_path)
        
        # 确保是RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 1. 随机亮度调整
        brightness_factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # 2. 随机对比度调整
        contrast_factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # 3. 随机添加高斯噪声
        noise_level = random.uniform(*noise_range)
        if noise_level > 0:
            # 转换为numpy数组
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # 添加高斯噪声
            noise = np.random.normal(0, noise_level, img_array.shape)
            noisy_img = img_array + noise
            
            # 限制到有效范围
            noisy_img = np.clip(noisy_img, 0, 1)
            
            # 转换回PIL图像
            image = Image.fromarray((noisy_img * 255).astype(np.uint8))
        
        # 4. 随机模糊处理
        blur_radius = random.uniform(*blur_range)
        if blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 保存图片
        image.save(output_path, quality=95, optimize=True)
        
        # 返回应用的参数
        params = {
            'brightness': brightness_factor,
            'contrast': contrast_factor,
            'noise': noise_level,
            'blur': blur_radius
        }
        
        return True, params
        
    except Exception as e:
        print(f"处理图片失败 {image_path}: {e}")
        return False, {}

def process_all_images(input_dir, output_dir, 
                      brightness_range=(0.8, 1.2),
                      contrast_range=(0.85, 1.15),
                      noise_range=(0.01, 0.03),
                      blur_range=(0, 1.5),
                      image_extensions=None):
    """
    处理目录下的所有图片
    """
    
    if image_extensions is None:
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已创建: {output_dir}")
    
    # 获取所有图片文件
    all_images = []
    for extension in image_extensions:
        pattern = os.path.join(input_dir, extension)
        all_images.extend(glob.glob(pattern, recursive=False))
        # 也匹配大写扩展名
        pattern_upper = os.path.join(input_dir, extension.upper())
        all_images.extend(glob.glob(pattern_upper, recursive=False))
    
    # 去重并排序
    all_images = sorted(list(set(all_images)))
    
    if not all_images:
        print(f"在目录 {input_dir} 中未找到任何图片文件")
        return
    
    print(f"找到 {len(all_images)} 张图片")
    print(f"亮度调整范围: {brightness_range[0]:.2f} - {brightness_range[1]:.2f}")
    print(f"对比度调整范围: {contrast_range[0]:.2f} - {contrast_range[1]:.2f}")
    print(f"噪声强度范围: {noise_range[0]:.3f} - {noise_range[1]:.3f}")
    print(f"模糊半径范围: {blur_range[0]:.1f} - {blur_range[1]:.1f}")
    print("开始处理...")
    
    # 统计信息
    success_count = 0
    failed_count = 0
    all_params = {
        'brightness': [],
        'contrast': [],
        'noise': [],
        'blur': []
    }
    
    # 处理每张图片
    for i, image_path in enumerate(all_images, 1):
        # 生成输出文件路径（保持相同的文件名）
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        # 应用随机增强
        success, params = apply_random_augmentations(
            image_path, output_path, 
            brightness_range, contrast_range, noise_range, blur_range
        )
        
        if success:
            success_count += 1
            
            # 收集参数统计
            for key in all_params:
                if key in params:
                    all_params[key].append(params[key])
            
            # 显示进度（每100张或前5张显示详情）
            if i <= 5 or i % 100 == 0 or i == len(all_images):
                print(f"[{i:4d}/{len(all_images)}] {filename}")
                print(f"    亮度: {params.get('brightness', 0):.3f}, "
                      f"对比度: {params.get('contrast', 0):.3f}, "
                      f"噪声: {params.get('noise', 0):.3f}, "
                      f"模糊: {params.get('blur', 0):.3f}")
            elif i % 50 == 0:
                print(f"[{i:4d}/{len(all_images)}] 已处理...")
                
        else:
            failed_count += 1
    
    # 输出统计结果
    print(f"\n=== 处理完成 ===")
    print(f"总图片数: {len(all_images)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {failed_count}")
    
    # 详细统计
    if all_params['brightness']:
        print(f"\n=== 增强参数统计 ===")
        
        # 亮度统计
        brightness_vals = all_params['brightness']
        print(f"亮度 - 平均: {np.mean(brightness_vals):.3f}, "
              f"范围: {np.min(brightness_vals):.3f} - {np.max(brightness_vals):.3f}")
        darker_count = sum(1 for f in brightness_vals if f < 1.0)
        brighter_count = sum(1 for f in brightness_vals if f > 1.0)
        print(f"    变暗: {darker_count} ({darker_count/len(brightness_vals)*100:.1f}%), "
              f"变亮: {brighter_count} ({brighter_count/len(brightness_vals)*100:.1f}%)")
        
        # 对比度统计
        contrast_vals = all_params['contrast']
        print(f"对比度 - 平均: {np.mean(contrast_vals):.3f}, "
              f"范围: {np.min(contrast_vals):.3f} - {np.max(contrast_vals):.3f}")
        
        # 噪声统计
        noise_vals = all_params['noise']
        print(f"噪声 - 平均: {np.mean(noise_vals):.4f}, "
              f"范围: {np.min(noise_vals):.4f} - {np.max(noise_vals):.4f}")
        
        # 模糊统计
        blur_vals = all_params['blur']
        print(f"模糊 - 平均: {np.mean(blur_vals):.3f}, "
              f"范围: {np.min(blur_vals):.3f} - {np.max(blur_vals):.3f}")
        no_blur_count = sum(1 for f in blur_vals if f == 0)
        print(f"    无模糊: {no_blur_count} ({no_blur_count/len(blur_vals)*100:.1f}%), "
              f"有模糊: {len(blur_vals)-no_blur_count} ({(len(blur_vals)-no_blur_count)/len(blur_vals)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='对图片进行随机多重增强（亮度+对比度+噪声+模糊）')
    parser.add_argument('--input', '-i', 
                       default='/home/baoadmin/LLM/WSC/dataset_video/extracted_frames',
                       help='输入图片目录路径')
    parser.add_argument('--output', '-o',
                       default='/home/baoadmin/LLM/WSC/dataset_video/extracted_frames_light',
                       help='输出图片目录路径')
    
    # 亮度参数
    parser.add_argument('--min-brightness', type=float, default=0.8,
                       help='最小亮度系数 (默认: 0.8)')
    parser.add_argument('--max-brightness', type=float, default=1.2,
                       help='最大亮度系数 (默认: 1.2)')
    
    # 对比度参数
    parser.add_argument('--min-contrast', type=float, default=0.85,
                       help='最小对比度系数 (默认: 0.85)')
    parser.add_argument('--max-contrast', type=float, default=1.15,
                       help='最大对比度系数 (默认: 1.15)')
    
    # 噪声参数
    parser.add_argument('--min-noise', type=float, default=0.01,
                       help='最小噪声强度 (默认: 0.01)')
    parser.add_argument('--max-noise', type=float, default=0.03,
                       help='最大噪声强度 (默认: 0.03)')
    
    # 模糊参数
    parser.add_argument('--min-blur', type=float, default=0,
                       help='最小模糊半径 (默认: 0)')
    parser.add_argument('--max-blur', type=float, default=1.5,
                       help='最大模糊半径 (默认: 1.5)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=None,
                       help='随机数种子，用于可重现的结果')
    
    # 预设模式
    parser.add_argument('--mode', choices=['conservative', 'moderate', 'aggressive'], 
                       default='moderate',
                       help='增强强度模式')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    # 根据模式调整参数
    if args.mode == 'conservative':
        brightness_range = (0.9, 1.1)
        contrast_range = (0.9, 1.1)
        noise_range = (0.005, 0.02)
        blur_range = (0, 1.0)
        print("使用保守增强模式")
    elif args.mode == 'aggressive':
        brightness_range = (0.7, 1.3)
        contrast_range = (0.8, 1.2)
        noise_range = (0.02, 0.05)
        blur_range = (0, 2.5)
        print("使用激进增强模式")
    else:  # moderate
        brightness_range = (args.min_brightness, args.max_brightness)
        contrast_range = (args.min_contrast, args.max_contrast)
        noise_range = (args.min_noise, args.max_noise)
        blur_range = (args.min_blur, args.max_blur)
        print("使用中等增强模式")
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在 - {args.input}")
        return
    
    if not os.path.isdir(args.input):
        print(f"错误: 输入路径不是目录 - {args.input}")
        return
    
    print("=== 图片多重随机增强工具 ===")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print()
    
    # 处理图片
    process_all_images(args.input, args.output, 
                      brightness_range, contrast_range, 
                      noise_range, blur_range)

if __name__ == '__main__':
    main()