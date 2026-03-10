#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为数据集添加GroundingDino检测信息的脚本
"""
import json
import os
import torch
from PIL import Image
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection
import copy
from tqdm import tqdm

# 安全关键词检测列表
SAFETY_KEYWORDS = [
    "Person", 
    "Lifting signal", 
    "Helmet", 
    "Long hair", 
    "Guardrail", 
    "Mobile phone", 
    "Electronic scale", 
    "Conveyor belt", 
    "Safety harness", 
    "Safety shoes", 
    "Cigarette",
    "face mask",
    "truck",
    "Crane",
]

# GroundingDino模型路径
GROUNDING_MODEL_PATH = r"/home/baoadmin/LLM/model/fushh7/llmdet_swin_large_hf"

class DatasetEnhancer:
    def __init__(self, grounding_model_path=GROUNDING_MODEL_PATH):
        """初始化数据集增强器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载GroundingDino模型
        print(f"加载GroundingDino模型从: {grounding_model_path}")
        self.processor = GroundingDinoProcessor.from_pretrained(grounding_model_path)
        self.grounding_model = GroundingDinoForObjectDetection.from_pretrained(grounding_model_path).to(self.device)
        
        # 构建文本查询
        self.text_query = ". ".join(SAFETY_KEYWORDS) + "."
        print(f"检测关键词: {', '.join(SAFETY_KEYWORDS)}")

    def detect_objects(self, image_path):
        """使用GroundingDino检测对象"""
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                return [], None
            
            # 加载图片
            image = Image.open(image_path)
            image_size = image.size  # (width, height)
            
            # 使用GroundingDino处理
            inputs = self.processor(images=image, text=self.text_query, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.35,
                text_threshold=0.35,
                target_sizes=[image.size[::-1]]  # Height, Width
            )
            
            # 提取检测结果
            result = results[0]
            detections = []
            
            if 'text_labels' in result and len(result['text_labels']) > 0:
                for box, text_label, score in zip(result["boxes"], result["text_labels"], result["scores"]):
                    # 找到匹配的关键词
                    matching_keyword = None
                    for keyword in SAFETY_KEYWORDS:
                        if keyword.lower() in text_label.lower() or text_label.lower() in keyword.lower():
                            matching_keyword = keyword
                            break
                    
                    if matching_keyword is None:
                        matching_keyword = text_label
                    
                    # 转换边界框格式
                    bbox = box.tolist()
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 确定位置描述
                    if center_x < image_size[0] / 3:
                        h_pos = "左侧"
                    elif center_x > image_size[0] * 2 / 3:
                        h_pos = "右侧"
                    else:
                        h_pos = "中部"
                    
                    if center_y < image_size[1] / 3:
                        v_pos = "上方"
                    elif center_y > image_size[1] * 2 / 3:
                        v_pos = "下方"
                    else:
                        v_pos = "中间"
                    
                    detection = {
                        "label": text_label,
                        "confidence": float(score),
                        "bbox": bbox,
                        "keyword": matching_keyword,
                        "position": f"图像{v_pos}{h_pos}",
                        "center": [center_x, center_y]
                    }
                    
                    detections.append(detection)
            
            return detections, image_size
            
        except Exception as e:
            print(f"检测对象时出错 ({image_path}): {str(e)}")
            return [], None

    def format_detection_info(self, detections, image_size, image_filename):
        """格式化检测信息为文本"""
        if not detections:
            return f"**{image_filename}：** 未检测到安全相关对象"
        
        # 按关键词分组
        grouped_detections = {}
        for det in detections:
            keyword = det["keyword"]
            if keyword not in grouped_detections:
                grouped_detections[keyword] = []
            grouped_detections[keyword].append(det)
        
        # 构建检测信息文本
        detection_info = f"**{image_filename} 检测结果：**\n"
        detection_info += f"图像尺寸: {image_size[0]}x{image_size[1]} 像素\n"
        detection_info += f"共检测到 {len(detections)} 个安全相关对象：\n"
        
        for keyword, objects in grouped_detections.items():
            # 中文名称映射
            keyword_mapping = {
                "Person": "人员", "Helmet": "安全帽", "Lifting signal": "吊装信号",
                "Long hair": "长发", "Guardrail": "护栏", "Mobile phone": "手机",
                "Electronic scale": "电子秤", "Conveyor belt": "输送带", 
                "Safety harness": "安全带", "Safety shoes": "安全鞋",
                "Cigarette": "香烟", "face mask": "口罩", "truck": "卡车", "Crane": "起重机/吊机"
            }
            
            keyword_cn = keyword_mapping.get(keyword, keyword)
            detection_info += f"• **{keyword_cn}** ({len(objects)}个):\n"
            
            for i, obj in enumerate(objects, 1):
                confidence = obj["confidence"]
                position = obj["position"]
                detection_info += f"  - 对象{i}: {position}，置信度{confidence:.2f}\n"
        
        return detection_info

    def process_images(self, image_paths):
        """处理多张图像并返回检测信息"""
        all_detection_info = "**目标检测辅助信息：**\n以下是通过AI视觉检测系统自动识别的图像中安全相关对象的详细信息：\n\n"
        
        for i, img_path in enumerate(image_paths):
            print(f"检测图像 {i+1}: {img_path}")
            detections, image_size = self.detect_objects(img_path)
            
            image_filename = os.path.basename(img_path)
            frame_label = f"第{i+1}帧 ({image_filename})"
            
            if detections and image_size:
                detection_info = self.format_detection_info(detections, image_size, frame_label)
                all_detection_info += detection_info + "\n\n"
            else:
                all_detection_info += f"**{frame_label}：** 未检测到安全相关对象\n\n"
        
        return all_detection_info.strip()

    def enhance_dataset(self, dataset_path, output_path=None):
        """增强数据集，添加检测信息"""
        # 读取原始数据集
        print(f"读取数据集: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果没有指定输出路径，则在原文件名基础上添加后缀
        if output_path is None:
            base_name = os.path.splitext(dataset_path)[0]
            output_path = f"{base_name}_enhanced.json"
        
        print(f"开始处理数据集，共 {len(data)} 条记录")
        
        # 处理每条记录
        enhanced_data = []
        for idx, item in enumerate(tqdm(data, desc="处理进度")):
            try:
                enhanced_item = copy.deepcopy(item)
                
                # 检查是否有images字段
                if 'images' not in item:
                    print(f"警告: 第 {idx+1} 条记录没有images字段，跳过")
                    enhanced_data.append(enhanced_item)
                    continue
                
                image_paths = item['images']
                if not image_paths:
                    print(f"警告: 第 {idx+1} 条记录的images字段为空，跳过")
                    enhanced_data.append(enhanced_item)
                    continue
                
                # 进行目标检测
                detection_info = self.process_images(image_paths)
                
                # 找到assistant的回复并添加检测信息
                messages = enhanced_item.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        original_content = msg.get('content', '')
                        # 在原内容前添加检测信息
                        enhanced_content = f"{detection_info}\n\n{original_content}"
                        msg['content'] = enhanced_content
                        break
                
                enhanced_data.append(enhanced_item)
                
            except Exception as e:
                print(f"处理第 {idx+1} 条记录时出错: {str(e)}")
                # 出错时保留原记录
                enhanced_data.append(item)
        
        # 保存增强后的数据集
        print(f"保存增强后的数据集到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据集增强完成！")
        print(f"原始记录数: {len(data)}")
        print(f"增强记录数: {len(enhanced_data)}")
        
        return output_path

def main():
    print("\n" + "=" * 60)
    print("数据集检测信息增强工具")
    print("=" * 60)
    
    # 配置参数
    dataset_path = "/home/baoadmin/LLM/WSC/dataset0611_all.json"
    output_path = "/home/baoadmin/LLM/WSC/dataset0611_all_enhanced.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件不存在: {dataset_path}")
        return
    
    # 初始化增强器
    try:
        enhancer = DatasetEnhancer()
    except Exception as e:
        print(f"初始化模型失败: {str(e)}")
        return
    
    # 增强数据集
    try:
        result_path = enhancer.enhance_dataset(dataset_path, output_path)
        print(f"\n✓ 数据集增强成功！")
        print(f"增强后的文件保存在: {result_path}")
    except Exception as e:
        print(f"数据集增强失败: {str(e)}")

if __name__ == "__main__":
    main()