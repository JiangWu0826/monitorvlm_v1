import os
import glob
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class YOLOPersonMaskProcessor:
    """精简版YOLOv8人体检测与掩码处理器"""
    
    def __init__(self, model_path, expand_factor=1.3, confidence_threshold=0.5, expand_method="ratio"):
        """
        初始化处理器
        
        Args:
            model_path (str): 本地模型权重路径
            expand_factor (float): BOX扩展系数（仅在ratio模式下使用）
            confidence_threshold (float): 检测置信度阈值
            expand_method (str): 扩展方式 "ratio"/"pixels"/"adaptive"
        """
        self.expand_factor = expand_factor
        self.confidence_threshold = confidence_threshold
        self.expand_method = expand_method
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载YOLO模型
        print(f"正在加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to('cuda')
        print("✓ YOLO模型加载完成，使用GPU")
    
    def detect_persons(self, image):
        """检测图像中的人体"""
        results = self.model(image, classes=[0], verbose=False, device='cuda')
        
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence >= self.confidence_threshold:
                        persons.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
        return persons
    
    def uniform_expand_bbox(self, bbox, image_size, expand_method="ratio"):
        """
        均匀扩展边界框，确保每个人获得相同的周边环境
        
        Args:
            bbox: 原始边界框 [x1, y1, x2, y2]
            image_size: 图像尺寸 (width, height)
            expand_method: 扩展方式
                - "ratio": 按比例扩展（当前方法）
                - "pixels": 按固定像素扩展
                - "adaptive": 自适应扩展
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        if expand_method == "pixels":
            # 方案2：固定像素扩展 - 每个人周围固定扩展N像素
            expand_pixels = 50  # 可调整：周围扩展的像素数
            new_x1 = max(0, x1 - expand_pixels)
            new_y1 = max(0, y1 - expand_pixels)
            new_x2 = min(image_size[0], x2 + expand_pixels)
            new_y2 = min(image_size[1], y2 + expand_pixels)
            
        elif expand_method == "adaptive":
            # 方案3：自适应扩展 - 根据人体尺寸自适应调整
            min_expand = 30   # 最小扩展像素
            max_expand = 100  # 最大扩展像素
            # 根据BOX大小决定扩展像素（大人扩展多，小人扩展少）
            box_size = (w * h) ** 0.5  # BOX尺寸的平方根
            expand_pixels = min(max_expand, max(min_expand, box_size * 0.3))
            
            new_x1 = max(0, x1 - expand_pixels)
            new_y1 = max(0, y1 - expand_pixels)
            new_x2 = min(image_size[0], x2 + expand_pixels)
            new_y2 = min(image_size[1], y2 + expand_pixels)
            
        else:  # "ratio" - 默认按比例扩展
            # 方案1：比例扩展 - 每个人按相同比例扩展
            expand_pixels_w = w * (self.expand_factor - 1) / 2
            expand_pixels_h = h * (self.expand_factor - 1) / 2
            
            new_x1 = max(0, x1 - expand_pixels_w)
            new_y1 = max(0, y1 - expand_pixels_h)
            new_x2 = min(image_size[0], x2 + expand_pixels_w)
            new_y2 = min(image_size[1], y2 + expand_pixels_h)
        
        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
    
    def compute_union_bbox(self, bboxes):
        """计算多个边界框的并集"""
        if not bboxes:
            return None
        
        min_x1 = min(bbox[0] for bbox in bboxes)
        min_y1 = min(bbox[1] for bbox in bboxes)
        max_x2 = max(bbox[2] for bbox in bboxes)
        max_y2 = max(bbox[3] for bbox in bboxes)
        
        return [min_x1, min_y1, max_x2, max_y2]
    
    def create_masked_image(self, image, persons):
        """创建掩码图像"""
        if not persons:
            return None
        
        img_array = np.array(image)
        image_size = image.size
        
        # 扩展所有人体边界框
        expanded_bboxes = []
        for person in persons:
            expanded_bbox = self.uniform_expand_bbox(person['bbox'], image_size, self.expand_method)
            expanded_bboxes.append(expanded_bbox)
        
        # 计算并集
        union_bbox = self.compute_union_bbox(expanded_bboxes)
        if union_bbox is None:
            return None
        
        # 创建掩码
        mask = np.zeros(img_array.shape[:2], dtype=bool)
        for bbox in expanded_bboxes:
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = True
        
        # 应用掩码
        masked_array = img_array.copy()
        masked_array[~mask] = [0, 0, 0] if len(img_array.shape) == 3 else 0
        
        # 裁剪到并集区域
        x1, y1, x2, y2 = union_bbox
        cropped_array = masked_array[y1:y2, x1:x2]
        
        return Image.fromarray(cropped_array)
    
    def process_single_image(self, img_path, output_dir):
        """处理单张图像"""
        try:
            image = Image.open(img_path)
            persons = self.detect_persons(image)
            
            if not persons:
                print(f"跳过 {Path(img_path).name}: 未检测到人体")
                return False
            
            masked_image = self.create_masked_image(image, persons)
            if masked_image is None:
                print(f"跳过 {Path(img_path).name}: 掩码创建失败")
                return False
            
            # 保存结果
            base_name = Path(img_path).stem
            ext = Path(img_path).suffix
            output_path = os.path.join(output_dir, f"{base_name}_M{ext}")
            masked_image.save(output_path, quality=95)
            
            print(f"✓ {Path(img_path).name} -> {base_name}_M{ext} (检测到 {len(persons)} 个人体)")
            return True
            
        except Exception as e:
            print(f"✗ {Path(img_path).name}: 处理失败 - {e}")
            return False
    
    def process_images_batch(self, image_paths, output_dir, num_workers=6):
        """批量处理图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        total = len(image_paths)
        processed = 0
        
        print(f"开始处理 {total} 张图像，使用 {num_workers} 个线程...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_single_image, img_path, output_dir) 
                      for img_path in image_paths]
            
            for future in futures:
                if future.result():
                    processed += 1
        
        total_time = time.time() - start_time
        print(f"\n处理完成: {processed}/{total} 张图像")
        print(f"总用时: {total_time:.2f} 秒")
        print(f"平均: {total_time/total:.3f} 秒/张")


def find_image_files(directory):
    """查找目录中的所有图像文件"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)


def main():
    # ========= 配置参数 =========
    INPUT_DIR = "/home/baoadmin/LLM/WSC/dataset_video/extracted_frames"
    OUTPUT_DIR = "/home/baoadmin/LLM/VISION_MASK/MASK_frame"
    MODEL_PATH = "/home/baoadmin/LLM/VISION_MASK/yolov8x.pt"  # 本地权重路径
    
    EXPAND_FACTOR = 6        # BOX扩展系数（ratio模式使用）
    CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
    NUM_WORKERS = 6             # 并行线程数
    
    # BOX扩展方式选择：
    EXPAND_METHOD = "pixels"    # "ratio": 按比例扩展（原方法）
                               # "pixels": 固定像素扩展（推荐）
                               # "adaptive": 自适应扩展
    # ==============================
    
    print("YOLOv8x人体检测与掩码处理")
    print("=" * 50)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"扩展方式: {EXPAND_METHOD}")
    if EXPAND_METHOD == "ratio":
        print(f"扩展系数: {EXPAND_FACTOR}")
    elif EXPAND_METHOD == "pixels":
        print(f"固定扩展: 50像素")
    else:
        print(f"自适应扩展: 30-100像素")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print("=" * 50)
    
    # 查找图像文件
    image_files = find_image_files(INPUT_DIR)
    if not image_files:
        print("❌ 未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 初始化处理器并开始处理
    try:
        processor = YOLOPersonMaskProcessor(MODEL_PATH, EXPAND_FACTOR, CONFIDENCE_THRESHOLD, EXPAND_METHOD)
        processor.process_images_batch(image_files, OUTPUT_DIR, NUM_WORKERS)
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")


if __name__ == "__main__":
    main()