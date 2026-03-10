import os
import torch
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection

# GroundingDino模型路径
GROUNDING_MODEL_PATH = r"/data/LLM/model/model/fushh7/llmdet_swin_large_hf"

# 只检测人员
PERSON_KEYWORDS = ["person"]

class PersonDetectionAndMask:
    def __init__(self, grounding_model_path=GROUNDING_MODEL_PATH):
        """初始化检测和掩码处理器"""
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载GroundingDino模型
        print(f"加载GroundingDino模型从: {grounding_model_path}")
        self.processor = GroundingDinoProcessor.from_pretrained(grounding_model_path)
        self.grounding_model = GroundingDinoForObjectDetection.from_pretrained(grounding_model_path).to(self.device)
        
        # 构建人员检测查询
        self.text_query = ". ".join(PERSON_KEYWORDS) + "."
        print(f"人员检测关键词: {', '.join(PERSON_KEYWORDS)}")

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU (Intersection over Union)"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算两个框的面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 计算并集面积
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def remove_duplicate_detections(self, persons, iou_threshold=0.5):
        """去除重复检测，使用IoU阈值进行过滤"""
        if len(persons) <= 1:
            return persons
        
        # 按置信度降序排序
        persons_sorted = sorted(persons, key=lambda x: x['confidence'], reverse=True)
        
        filtered_persons = []
        
        for current_person in persons_sorted:
            is_duplicate = False
            current_bbox = current_person['bbox']
            
            # 检查当前检测是否与已保留的检测重复
            for kept_person in filtered_persons:
                kept_bbox = kept_person['bbox']
                iou = self.calculate_iou(current_bbox, kept_bbox)
                
                if iou > iou_threshold:
                    is_duplicate = True
                    print(f"检测到重复: Person {current_person['id']} 与 Person {kept_person['id']} IoU={iou:.3f}")
                    break
            
            if not is_duplicate:
                filtered_persons.append(current_person)
        
        # 重新分配ID
        for i, person in enumerate(filtered_persons):
            person['id'] = i + 1
        
        return filtered_persons

    def detect_persons(self, image_path):
        """使用GroundingDino检测人员"""
        try:
            # 加载图片
            image = Image.open(image_path)
            image_size = image.size  # (width, height)
            
            # 使用GroundingDino处理
            inputs = self.processor(images=image, text=self.text_query, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            
            # 提高阈值以减少误检测和重复检测
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.35,  # 提高box阈值从0.25到0.35
                text_threshold=0.30,  # 提高text阈值从0.25到0.30
                target_sizes=[image.size[::-1]]  # Height, Width
            )
            
            # 提取检测结果
            result = results[0]
            persons = []
            
            if 'text_labels' in result and len(result['text_labels']) > 0:
                for i, (box, text_label, score) in enumerate(zip(result["boxes"], result["text_labels"], result["scores"])):
                    # 只保留person标签的检测结果
                    if text_label.lower() not in ['person', 'people', 'man', 'woman', 'human']:
                        continue
                    
                    # 转换边界框格式
                    bbox = box.tolist()
                    x1, y1, x2, y2 = bbox
                    
                    # 检查边界框是否合理
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # 过滤掉太小的检测框（可能是误检测）
                    min_box_area = image_size[0] * image_size[1] * 0.001  # 图像面积的0.1%
                    if box_area < min_box_area:
                        print(f"过滤掉过小的检测框: 面积={box_area:.0f}, 最小要求={min_box_area:.0f}")
                        continue
                    
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
                    
                    person = {
                        "id": i + 1,
                        "label": text_label,
                        "confidence": float(score),
                        "bbox": bbox,
                        "position": f"图像{v_pos}{h_pos}",
                        "center": [center_x, center_y]
                    }
                    
                    persons.append(person)
                
                print(f"初始检测到 {len(persons)} 个人员")
                
                # 去除重复检测
                persons = self.remove_duplicate_detections(persons, iou_threshold=0.5)
                print(f"去重后保留 {len(persons)} 个人员")
            
            return persons, image, image_size
            
        except Exception as e:
            print(f"检测人员时出错: {str(e)}")
            return [], None, None

    def create_masked_image(self, image, persons, save_dir=None, add_margin=True):
        """创建掩码图像：保留检测框内容，其他区域变为黑色"""
        try:
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 创建黑色背景图像
            masked_image = Image.new('RGB', image.size, (0, 0, 0))
            
            # 将原图转换为numpy数组以便处理
            original_array = np.array(image)
            masked_array = np.array(masked_image)
            
            # 为每个检测到的人员创建掩码
            for person in persons:
                bbox = person["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                if add_margin:
                    # 增大边界框的边距，使框更大
                    box_width = x2 - x1
                    box_height = y2 - y1
                    margin_x = int(box_width * 0.3)  # 宽度的30%作为边距
                    margin_y = int(box_height * 0.3)  # 高度的30%作为边距
                    
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(image.size[0], x2 + margin_x)
                    y2 = min(image.size[1], y2 + margin_y)
                
                # 将检测框内的原图内容复制到掩码图像
                masked_array[y1:y2, x1:x2] = original_array[y1:y2, x1:x2]
                
                print(f"Person {person['id']} 掩码区域: ({x1}, {y1}) -> ({x2}, {y2})")
            
            # 转换回PIL图像
            masked_image = Image.fromarray(masked_array)
            
            # 保存掩码图像
            if save_dir:
                masked_path = os.path.join(save_dir, "masked_persons.jpg")
                masked_image.save(masked_path, quality=95)
                print(f"✅ 掩码图像已保存: {masked_path}")
                return masked_path, masked_image
            
            return None, masked_image
            
        except Exception as e:
            print(f"❌ 创建掩码图像时出错: {str(e)}")
            return None, None

    def create_individual_masks(self, image, persons, save_dir=None):
        """为每个检测到的人员创建单独的掩码图像"""
        individual_masks = []
        
        try:
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            original_array = np.array(image)
            
            for person in persons:
                # 创建黑色背景图像
                masked_image = Image.new('RGB', image.size, (0, 0, 0))
                masked_array = np.array(masked_image)
                
                bbox = person["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # 增大边界框的边距
                box_width = x2 - x1
                box_height = y2 - y1
                margin_x = int(box_width * 0.3)
                margin_y = int(box_height * 0.3)
                
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(image.size[0], x2 + margin_x)
                y2 = min(image.size[1], y2 + margin_y)
                
                # 将该人员区域的原图内容复制到掩码图像
                masked_array[y1:y2, x1:x2] = original_array[y1:y2, x1:x2]
                
                # 转换回PIL图像
                individual_mask = Image.fromarray(masked_array)
                
                # 保存单独的掩码图像
                if save_dir:
                    mask_path = os.path.join(save_dir, f"person_{person['id']}_mask.jpg")
                    individual_mask.save(mask_path, quality=95)
                    print(f"✅ Person {person['id']} 掩码图像已保存: {mask_path}")
                    person["mask_path"] = mask_path
                
                person["masked_image"] = individual_mask
                individual_masks.append(person)
            
            return individual_masks
            
        except Exception as e:
            print(f"❌ 创建单独掩码图像时出错: {str(e)}")
            return []

    def draw_detection_results(self, image, persons, save_dir):
        """在原图上绘制检测框"""
        try:
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # 设置字体（如果系统有的话）
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
            
            for i, person in enumerate(persons):
                bbox = person["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # 使用增大后的框进行绘制
                box_width = x2 - x1
                box_height = y2 - y1
                margin_x = int(box_width * 0.3)
                margin_y = int(box_height * 0.3)
                
                x1_expanded = max(0, x1 - margin_x)
                y1_expanded = max(0, y1 - margin_y)
                x2_expanded = min(image.size[0], x2 + margin_x)
                y2_expanded = min(image.size[1], y2 + margin_y)
                
                color = colors[i % len(colors)]
                
                # 绘制边界框
                draw.rectangle([x1_expanded, y1_expanded, x2_expanded, y2_expanded], 
                             outline=color, width=3)
                
                # 绘制标签
                label = f"Person {person['id']} ({person['confidence']:.2f})"
                draw.text((x1_expanded, y1_expanded - 25), label, fill=color, font=font)
            
            # 保存标注图像
            detection_path = os.path.join(save_dir, "detection_results.jpg")
            draw_image.save(detection_path, quality=95)
            print(f"✅ 检测结果图像已保存: {detection_path}")
            
            return detection_path
            
        except Exception as e:
            print(f"❌ 绘制检测结果时出错: {str(e)}")
            return None

def main():
    """人员检测+掩码处理主函数"""
    print("\n" + "=" * 60)
    print("【人员检测与掩码处理】")
    print("=" * 60)
    
    # 统一测试图片路径
    image_path = "/data/LLM/WSC/dataset_video/extracted_frames/81-15.jpg"  # 请输入图像路径
    save_dir = "/data/LLM/wjj/模型推理/视频测试结果"  # 请输入保存目录路径
    
    # 初始化分析器
    analyzer = PersonDetectionAndMask()
    print("✅ 人员检测器初始化完成")
    
    # 检测人员
    persons, image, image_size = analyzer.detect_persons(image_path)
    
    if not persons:
        print("❌ 未检测到人员")
        return
    
    print(f"✅ 最终检测到 {len(persons)} 个人员")
    for person in persons:
        print(f"   - Person {person['id']}: {person['position']}, 置信度: {person['confidence']:.3f}")
    
    # 绘制检测结果
    analyzer.draw_detection_results(image, persons, save_dir)
    
    # 创建统一掩码图像（所有人员在一张图上）
    masked_path, masked_image = analyzer.create_masked_image(image, persons, save_dir)
    
    if masked_path:
        print(f"🎉 成功创建统一掩码图像: {masked_path}")
    
    # 可选：创建每个人员的单独掩码图像
    print("\n" + "-" * 40)
    print("创建单独掩码图像...")
    individual_masks = analyzer.create_individual_masks(image, persons, save_dir)
    
    if individual_masks:
        print(f"✅ 已创建 {len(individual_masks)} 个单独掩码图像")
    else:
        print("❌ 单独掩码图像创建失败")

if __name__ == "__main__":
    main()