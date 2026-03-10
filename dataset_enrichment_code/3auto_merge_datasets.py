#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动扫描合并数据集脚本
自动扫描指定目录下的所有JSON/JSONL文件并合并
"""

import json
import os
import glob
import argparse
from pathlib import Path

def find_json_files(directory, pattern="*.json", recursive=False):
    """
    自动扫描目录下的JSON文件
    
    Args:
        directory: 目录路径
        pattern: 文件名模式 (如 "*.json", "safety_*.json")
        recursive: 是否递归搜索子目录
    
    Returns:
        list: JSON文件路径列表
    """
    json_files = []
    
    if recursive:
        # 递归搜索
        search_pattern = os.path.join(directory, "**", pattern)
        json_files = glob.glob(search_pattern, recursive=True)
    else:
        # 只搜索当前目录
        search_pattern = os.path.join(directory, pattern)
        json_files = glob.glob(search_pattern)
    
    # 也搜索JSONL文件
    if pattern == "*.json":
        jsonl_pattern = pattern.replace(".json", ".jsonl")
        if recursive:
            search_pattern = os.path.join(directory, "**", jsonl_pattern)
            json_files.extend(glob.glob(search_pattern, recursive=True))
        else:
            search_pattern = os.path.join(directory, jsonl_pattern)
            json_files.extend(glob.glob(search_pattern))
    
    # 排序并去重
    json_files = sorted(list(set(json_files)))
    
    return json_files

def merge_datasets(input_files, output_file, output_format='json'):
    """
    合并多个数据集文件
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
        output_format: 输出格式 ('json' 或 'jsonl')
    """
    
    print("=" * 60)
    print("自动数据集合并工具")
    print("=" * 60)
    
    all_data = []
    successful_files = 0
    failed_files = 0
    
    # 读取所有输入文件
    for i, input_file in enumerate(input_files, 1):
        print(f"\n处理文件 {i}/{len(input_files)}: {os.path.basename(input_file)}")
        print(f"  路径: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"  ✗ 文件不存在，跳过")
            failed_files += 1
            continue
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(input_file)
            print(f"  文件大小: {file_size:,} 字节")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                # 检查文件格式
                if input_file.endswith('.json'):
                    # JSON格式
                    data = json.load(f)
                    if isinstance(data, list):
                        records_count = len(data)
                        all_data.extend(data)
                    else:
                        # 单个记录
                        records_count = 1
                        all_data.append(data)
                
                elif input_file.endswith('.jsonl'):
                    # JSONL格式
                    records_count = 0
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                all_data.append(record)
                                records_count += 1
                            except json.JSONDecodeError as e:
                                print(f"  ⚠ 第{line_num}行JSON格式错误: {e}")
                                continue
                
                else:
                    # 尝试自动检测格式
                    f.seek(0)
                    content = f.read().strip()
                    
                    # 尝试作为JSON解析
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            records_count = len(data)
                            all_data.extend(data)
                        else:
                            records_count = 1
                            all_data.append(data)
                    except:
                        # 尝试作为JSONL解析
                        records_count = 0
                        for line_num, line in enumerate(content.split('\n'), 1):
                            line = line.strip()
                            if line:
                                try:
                                    record = json.loads(line)
                                    all_data.append(record)
                                    records_count += 1
                                except:
                                    continue
            
            print(f"  ✓ 成功读取 {records_count:,} 条记录")
            successful_files += 1
            
        except Exception as e:
            print(f"  ✗ 读取文件失败: {e}")
            failed_files += 1
            continue
    
    # 写入合并后的文件
    print(f"\n" + "="*60)
    print(f"合并摘要:")
    print(f"  成功处理文件: {successful_files}")
    print(f"  失败文件: {failed_files}")
    print(f"  总记录数: {len(all_data):,}")
    print(f"\n开始写入合并文件: {output_file}")
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if output_format.lower() == 'json':
                # 输出为JSON格式
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            else:
                # 输出为JSONL格式
                for record in all_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # 检查输出文件大小
        output_size = os.path.getsize(output_file)
        
        print(f"✓ 合并完成!")
        print(f"  输出文件: {output_file}")
        print(f"  输出格式: {output_format.upper()}")
        print(f"  文件大小: {output_size:,} 字节")
        print(f"  总记录数: {len(all_data):,}")
        
        return output_file
        
    except Exception as e:
        print(f"✗ 写入文件失败: {e}")
        return None

def validate_merged_file(merged_file):
    """
    验证合并后的文件
    """
    print(f"\n验证合并文件: {os.path.basename(merged_file)}")
    
    try:
        if merged_file.endswith('.json'):
            # JSON格式验证
            with open(merged_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  ✓ JSON格式正确")
                print(f"  记录数: {len(data):,}")
                
                # 检查第一条记录的字段
                if data:
                    first_record = data[0]
                    print(f"  第一条记录字段: {list(first_record.keys())}")
                    
                    # 检查messages字段结构
                    if 'messages' in first_record and first_record['messages']:
                        msg_count = len(first_record['messages'])
                        print(f"  第一条记录消息数: {msg_count}")
                        
                        # 检查消息类型分布
                        roles = [msg.get('role', 'unknown') for msg in first_record['messages']]
                        role_counts = {role: roles.count(role) for role in set(roles)}
                        print(f"  消息角色分布: {role_counts}")
                    
                    # 检查图片字段
                    if 'images' in first_record:
                        print(f"  第一条记录图片数: {len(first_record['images'])}")
                        
                        # 抽样检查图片文件
                        if first_record['images']:
                            sample_image = first_record['images'][0]
                            if os.path.exists(sample_image):
                                print(f"  ✓ 图片文件存在 (抽样检查)")
                            else:
                                print(f"  ⚠ 图片文件不存在: {sample_image}")
        
        elif merged_file.endswith('.jsonl'):
            # JSONL格式验证
            record_count = 0
            sample_record = None
            
            with open(merged_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            record_count += 1
                            if sample_record is None:
                                sample_record = record
                        except Exception as e:
                            print(f"  ✗ 第{line_num}行格式错误: {e}")
                            return False
            
            print(f"  ✓ JSONL格式正确")
            print(f"  记录数: {record_count:,}")
            
            if sample_record:
                print(f"  第一条记录字段: {list(sample_record.keys())}")
                
                # 检查图片字段
                if 'images' in sample_record:
                    print(f"  第一条记录图片数: {len(sample_record['images'])}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        return False

def analyze_dataset_info(merged_file):
    """
    分析合并后数据集的详细信息
    """
    print(f"\n分析数据集详细信息...")
    
    try:
        records = []
        
        if merged_file.endswith('.json'):
            with open(merged_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
        elif merged_file.endswith('.jsonl'):
            with open(merged_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        
        if not records:
            print("  无数据记录")
            return
        
        # 统计信息
        total_images = 0
        text_lengths = []
        image_paths = set()
        
        for record in records:
            # 统计图片数量
            if 'images' in record and record['images']:
                total_images += len(record['images'])
                # 收集图片路径用于去重统计
                for img_path in record['images']:
                    image_paths.add(img_path)
            
            # 统计文本长度
            total_text = ""
            if 'messages' in record:
                for message in record['messages']:
                    content = message.get('content', '')
                    if isinstance(content, str):
                        total_text += content + " "
                    elif isinstance(content, list):
                        # 处理多模态内容
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                total_text += item['text'] + " "
            
            text_lengths.append(len(total_text))
        
        print(f"\n数据集统计信息:")
        print(f"  总记录数: {len(records):,}")
        print(f"  总图片引用数: {total_images:,}")
        print(f"  唯一图片数: {len(image_paths):,}")
        
        if total_images > 0:
            print(f"  平均每条记录图片数: {total_images/len(records):.1f}")
        
        if text_lengths:
            avg_length = sum(text_lengths)/len(text_lengths)
            print(f"  平均文本长度: {avg_length:.0f} 字符")
            print(f"  最长文本: {max(text_lengths):,} 字符")
            print(f"  最短文本: {min(text_lengths):,} 字符")
        
        # 抽样检查图片文件存在性
        if image_paths:
            sample_size = min(10, len(image_paths))
            sample_images = list(image_paths)[:sample_size]
            existing_count = sum(1 for img in sample_images if os.path.exists(img))
            print(f"  图片文件存在性 (抽样{sample_size}个): {existing_count}/{sample_size}")
        
    except Exception as e:
        print(f"  分析失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='自动扫描合并数据集工具')
    parser.add_argument('--input-dir', '-i', 
                       default='/home/baoadmin/LLM/WSC/data_train/混合数据集制作和生成',
                       help='输入目录路径')
    parser.add_argument('--output', '-o',
                       default='/home/baoadmin/LLM/WSC/expand_dataset/202506027dataset2.json',
                       help='输出文件路径')
    parser.add_argument('--pattern', '-p',
                       default='*.json',
                       help='文件名模式 (如 "safety_*.json", "*.json")')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='递归搜索子目录')
    parser.add_argument('--format', '-f', choices=['json', 'jsonl'], default='json',
                       help='输出格式')
    parser.add_argument('--no-validate', action='store_true',
                       help='跳过文件验证')
    parser.add_argument('--no-analyze', action='store_true',
                       help='跳过详细分析')
    
    args = parser.parse_args()
    
    print("自动数据集合并工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"文件模式: {args.pattern}")
    print(f"递归搜索: {'是' if args.recursive else '否'}")
    print(f"输出文件: {args.output}")
    print(f"输出格式: {args.format.upper()}")
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"\n错误: 输入目录不存在 - {args.input_dir}")
        return
    
    if not os.path.isdir(args.input_dir):
        print(f"\n错误: 输入路径不是目录 - {args.input_dir}")
        return
    
    # 自动扫描JSON文件
    print(f"\n扫描目录中...")
    json_files = find_json_files(args.input_dir, args.pattern, args.recursive)
    
    if not json_files:
        print(f"在目录 {args.input_dir} 中未找到匹配 '{args.pattern}' 的文件")
        return
    
    print(f"\n发现 {len(json_files)} 个文件:")
    for i, file_path in enumerate(json_files, 1):
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"  {i:2d}. {os.path.basename(file_path)} ({file_size:,} 字节)")
    
    # 询问是否继续
    response = input(f"\n是否继续合并这 {len(json_files)} 个文件? (y/n, 默认y): ").strip().lower()
    if response == 'n':
        print("已取消合并")
        return
    
    # 执行合并
    result = merge_datasets(json_files, args.output, args.format)
    
    if result:
        print("\n" + "=" * 60)
        
        # 验证结果
        if not args.no_validate:
            if validate_merged_file(result):
                print("✓ 文件验证通过")
            else:
                print("✗ 文件验证失败")
        
        # 分析数据集信息
        if not args.no_analyze:
            analyze_dataset_info(result)
        
        print("\n" + "=" * 60)
        print("✓ 合并完成！")
        print(f"输出文件: {result}")
    else:
        print("\n✗ 合并失败！")

if __name__ == "__main__":
    main()