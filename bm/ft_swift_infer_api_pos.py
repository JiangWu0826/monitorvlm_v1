#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合GroundingDino检测的安全分析脚本
"""
import requests
import base64
import json
import os
import time
import torch
from PIL import Image
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection
import os
import json
import glob


# 服务地址
BASE_URL = "http://localhost:8008/v1"

SAFETY_KEYWORDS = [
    "Person",
    "Vehicle", 
    "Grinding machine", 
    "Helmet", 
    "Iron chips", 
    "Forklift", 
    "Drill press", 
    "Electric tricycle", 
    "Loader", 
    "Equipment boom", 
    "safety belt", 
    "Fall arrestor", 
    "Lifting machinery", 
    "Edge", 
    "Hole", 
    "Safety shoes", 
    "Smoking", 
    "Face mask", 
    "Gloves", 
]


# GroundingDino模型路径
GROUNDING_MODEL_PATH = r"/home/baoadmin/LLM/model/fushh7/llmdet_swin_large_hf"

class IntegratedSafetyAnalyzer:
    def __init__(self, grounding_model_path=GROUNDING_MODEL_PATH):
        """初始化分析器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载GroundingDino模型
        print(f"加载GroundingDino模型从: {grounding_model_path}")
        self.processor = GroundingDinoProcessor.from_pretrained(grounding_model_path)
        self.grounding_model = GroundingDinoForObjectDetection.from_pretrained(grounding_model_path).to(self.device)
        
        # 构建文本查询
        self.text_query = ". ".join(SAFETY_KEYWORDS) + "."
        print(f"检测关键词: {', '.join(SAFETY_KEYWORDS)}")

    def encode_image(self, image_path):
        """图像转base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def detect_objects(self, image_path):
        """使用GroundingDino检测对象"""
        try:
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
            print(f"检测对象时出错: {str(e)}")
            return [], None

    def format_detection_info(self, detections, image_size):
        """格式化检测信息为文本"""
        if not detections:
            return "未检测到安全相关对象。"
        
        # 按关键词分组
        grouped_detections = {}
        for det in detections:
            keyword = det["keyword"]
            if keyword not in grouped_detections:
                grouped_detections[keyword] = []
            grouped_detections[keyword].append(det)
        
        # 构建检测信息文本
        detection_info = f"**检测到的安全对象信息：**\n"
        detection_info += f"图像尺寸: {image_size[0]}x{image_size[1]} 像素\n"
        detection_info += f"共检测到 {len(detections)} 个安全相关对象：\n\n"
        
        for keyword, objects in grouped_detections.items():
            # 中文名称映射
            keyword_mapping = {
                "Person": "人员", "Helmet": "安全帽", "Lifting signal": "吊装信号",
                "Long hair": "长发", "Guardrail": "护栏", "Mobile phone": "手机",
                "Electronic scale": "电子秤", "Conveyor belt": "输送带", 
                "Safety harness": "安全带", "Safety shoes": "安全鞋",
                "Cigarette": "香烟", "face mask": "口罩", "truck": "卡车",  "crane": "起重机/吊机"
            }
            
            keyword_cn = keyword_mapping.get(keyword, keyword)
            detection_info += f"• **{keyword_cn}** ({len(objects)}个):\n"
            
            for i, obj in enumerate(objects, 1):
                confidence = obj["confidence"]
                position = obj["position"]
                detection_info += f"  - 对象{i}: {position}，置信度{confidence:.2f}\n"
            
            detection_info += "\n"
        
        return detection_info

    def create_enhanced_system_prompt(self, detection_info):
        """创建包含检测信息的增强system prompt"""
        base_prompt = """你是一个专业的工业安全监控分析专家。你的任务是分析监控图像，识别违反安全规则的行为。

**重要原则：只有在图像中能清晰观察到下列的违规行为时才判定违规。许多监控图像可能合规，不要强行寻找违规行为。**

安全规则列表：
* 生产作业场所未佩戴安全帽的
* 作业时，女工作人员披肩长发未盘进安全帽内的
* 无安全措施的攀爬坠落高度1～2m的平台栏杆，或者从高处跳下来的
* 作业场所边行走边看手机的
* 皮带机不停机,工作人员在附近清扫、完成其他任务的
* 起重机械启动过程中 ，人员上下车或人员靠的很近 
* 临边作业不系安全带
* 在专业的实验室里面，工作人员没穿劳保鞋
* 在井下吸烟的
* 切割作业时，附近的工作人员必须戴口罩
* 作业人员保险带系挂不规范
* 运料车、运人车门未关闭就行驶的 

**判断标准：**
* 确认违反：图像中清晰可见违规行为，证据确凿
* 疑似违反：可能存在违规但图像不够清晰或存在其他合理解释
* 未违反：没有观察到违规行为或证据不足
* 当无法清楚判断时，倾向于判定为未违反

**目标检测辅助信息：**
以下是通过AI视觉检测系统自动识别的图像中安全相关对象的详细信息，请结合这些信息进行分析：

""" + detection_info + """

**分析要求：**
1. 请先详细分析图像内容，客观描述观察到的事实
2. 结合检测到的对象信息，逐项检查相关安全规则
3. 重点关注人员与安全设备的配对关系（如人员是否佩戴安全帽、安全带等）
4. 对于每个可能的违规点，评估证据充分性
5. 利用对象位置信息进行精确的空间关系分析"""

        return base_prompt

    def call_safety_analysis_with_detection(self, image_paths, user_query):
        """调用安全分析API（包含检测信息）"""
        # 对每张图片进行目标检测
        all_detection_info = ""
        
        for i, img_path in enumerate(image_paths):
            print(f"检测图像 {i+1}: {img_path}")
            detections, image_size = self.detect_objects(img_path)
            
            if detections:
                print(f"检测到 {len(detections)} 个对象")
                detection_info = self.format_detection_info(detections, image_size)
                all_detection_info += f"\n**图像 {i+1} ({os.path.basename(img_path)}) 检测结果：**\n"
                all_detection_info += detection_info + "\n"
            else:
                print(f"未检测到对象")
                all_detection_info += f"\n**图像 {i+1} ({os.path.basename(img_path)})：** 未检测到安全相关对象\n"
        
        # 创建增强的system prompt
        enhanced_system_prompt = self.create_enhanced_system_prompt(all_detection_info)
        
        # 准备图像内容
        content = [{"type": "text", "text": user_query}]
        
        for img_path in image_paths:
            base64_image = self.encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        # API请求
        payload = {
            "model": "Qwen2.5-VL-72B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system_prompt
                },
                {
                    "role": "user", 
                    "content": content
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            print("正在发送API请求...")
            response = requests.post(f"{BASE_URL}/chat/completions", 
                                   json=payload, 
                                   timeout=200)
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    print("响应格式异常")
                    return None
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"请求异常: {e}")
            return None

def main():
    print("\n" + "=" * 60)
    print("整合GroundingDino检测的安全分析系统")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = IntegratedSafetyAnalyzer()
    
    # 图像路径
    image_paths = [
        "/home/baoadmin/LLM/WSC/dataset_video/data_video/test_video/extrame_frame/4-10.jpg",
        "/home/baoadmin/LLM/WSC/dataset_video/data_video/test_video/extrame_frame/4-11.jpg", 
        "/home/baoadmin/LLM/WSC/dataset_video/data_video/test_video/extrame_frame/4-12.jpg"
    ]
    
    # 检查文件是否存在
    valid_images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            valid_images.append(img_path)
            print(f"✓ 找到图像: {img_path}")
        else:
            print(f"✗ 图像不存在: {img_path}")
    
    if not valid_images:
        print("没有找到有效的图像文件，请检查路径")
        return
    
    # 用户查询
    query = """<image><image><image>

请分析这三帧连续监控图像，结合目标检测信息进行安全分析，按以下格式回答：

**场景描述：**
[客观描述图像中的环境、人员、设备、作业活动等基本情况，结合检测到的对象信息]

**检测信息验证：**
[验证和补充目标检测结果，说明检测的准确性和遗漏情况]

**思维链：**
[详细分析过程]
* 相关规则识别：[列出在当前场景中适用的安全规则]
* 人员-设备配对分析：[分析检测到的人员与安全设备的对应关系]
* 逐项检查：
   * 规则X：[观察到的具体情况] → [结合检测信息的证据] → [是否构成违规] → [证据评估]
   * ...
* 综合评估：[基于证据评估得出的总体安全状况]

**违反的规则：**
* 确认违反：[列出确定违反的规则，如无则写"无"]
* 疑似违反：[列出可能但不确定的规则，如无则写"无"] 
* 未违反：[列出明确未违反的相关规则，如无适用规则则写"当前场景下所有相关规则均合规"]

**总体结论：** [安全合规 / 存在违规]

**改进建议：**
[基于检测结果和分析，提供具体的安全改进建议]"""
    
    # 开始分析
    start_time = time.time()
    result = analyzer.call_safety_analysis_with_detection(valid_images, query)
    analysis_time = time.time() - start_time
    
    if result:
        print("\n" + "=" * 80)
        print("整合分析结果:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        print(f"总分析时间: {analysis_time:.2f} 秒")
    else:
        print("分析失败")

if __name__ == "__main__":
    main()