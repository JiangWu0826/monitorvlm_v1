#!/usr/bin/env python3
"""
动态规则安全违规检测模型推理脚本 + 规则缩放测试
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import logging
import argparse
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import csv

# 设置日志和字体
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# ================================ 模型定义 ================================

class DynamicSafetyClassifier(nn.Module):
    """动态规则安全分类器 - 输入单条规则和图片，输出单个概率"""
    
    def __init__(self, hidden_dim=1024, preserve_aspect_ratio=True):
        super().__init__()
        
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # 1. 图像编码器 (ResNet50) - 冻结
        logger.info("🔧 加载ResNet50...")
        self.image_encoder = models.resnet50(weights='IMAGENET1K_V1')
        
        if preserve_aspect_ratio:
            self.image_encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.image_encoder.fc = nn.Identity()
        else:
            self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        
        # 冻结ResNet50的所有参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()
        
        # 2. 文本编码器 (Chinese Longformer) - 冻结
        logger.info("🔧 加载Chinese Longformer...")
        
        longformer_model_name = 'schen/longformer-chinese-base-4096'
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(longformer_model_name)
            self.text_encoder = AutoModel.from_pretrained(longformer_model_name)
            logger.info(f"✅ 成功加载Chinese Longformer: {longformer_model_name}")
        except Exception as e:
            logger.warning(f"⚠️ 无法加载 {longformer_model_name}, 使用备用模型...")
            longformer_model_name = 'bert-base-chinese'
            self.tokenizer = AutoTokenizer.from_pretrained(longformer_model_name)
            self.text_encoder = AutoModel.from_pretrained(longformer_model_name)
            logger.info(f"✅ 使用备用模型: {longformer_model_name}")
        
        # 冻结文本编码器的所有参数
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        
        # 3. 多模态融合分类器
        text_dim = self.text_encoder.config.hidden_size  # 通常是768
        input_dim = 2048 + text_dim  # ResNet + Text
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),          # 2816 -> 1024
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),    # 1024 -> 512
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 4, 1),             # 256 -> 1
            nn.Sigmoid()  # 输出0-1概率
        )
        
    def encode_rule_texts(self, rule_texts):
        """编码规则文本（batch处理）"""
        with torch.no_grad():
            # 批量编码文本
            encoded = self.tokenizer(
                rule_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到正确的设备
            device = next(self.text_encoder.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # 获取文本编码
            outputs = self.text_encoder(**encoded)
            text_features = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size] CLS token
            
        return text_features
        
    def forward(self, images, rule_texts):
        """
        前向传播
        Args:
            images: [B, 3, H, W] 图像tensor
            rule_texts: [B] 规则文本列表
        Returns:
            predictions: [B] 违规概率
        """
        batch_size = images.size(0)
        
        # 1. 图像特征提取
        with torch.no_grad():
            image_features = self.image_encoder(images)  # [B, 2048]
            image_features = image_features.view(batch_size, -1)
        
        # 2. 文本特征提取
        text_features = self.encode_rule_texts(rule_texts)  # [B, text_dim]
        
        # 3. 特征融合
        combined_features = torch.cat([image_features, text_features], dim=1)  # [B, 2048+text_dim]
        
        # 4. 分类预测
        predictions = self.classifier(combined_features)  # [B, 1]
        
        return predictions.squeeze(-1)  # [B] 返回标量概率

# ================================ 推理类 ================================

class SafetyRuleInference:
    """安全规则推理类"""
    
    def __init__(self, model_path, device='auto'):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径
            device: 设备 ('auto', 'cpu', 'cuda')
        """
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"🚀 使用设备: {self.device}")
        
        # 默认安全规则列表
        self.default_rules = [
            "车辆（人行车、无轨设备）行驶过程中，将身体伸出车外的",
            "砂轮机、切割机在切割时前方有人员的", 
            "清扫铁屑，直接用手去清扫的",
            "叉车所叉物件超宽未采取防护措施的",
            "台钻运行中用手或用布清除铁屑的",
            "井下设备、车辆无照明运行",
            "叉车载人的",
            "设备运转时工作人员隔着不同的机械，进行传递物件",
            "翻转大型物体时，倾倒方向投影面积内站人的",
            "攀爬作业未正确使用双钩安全带或防坠器的",
            "电动三轮车驾驶位置只能乘坐一个人，违规超载的",
            "使用角磨机未佩戴防护眼镜",
            "铲运机、铲车等停车时铲斗未落地",
            "高处进行抛物的",
            "设备大臂动作时，人员站在大臂动作覆盖范围内的",
            "生产作业场所未佩戴安全帽的",
            "作业时，女工作人员披肩长发未盘进安全帽内的",
            "工作人员无安全措施的攀爬坠落高度1～2m的平台栏杆，或者工作人员从50cm高处跳下来的",
            "工作场合看手机的",
            "皮带机不停机，工作人员在附近清扫、完成其他任务的",
            "起重机械启动过程中，人员上下车或人员靠的起重机械很近",
            "在临边、洞口处工作不系安全带",
            "在专业的实验室里面，工作人员没穿劳保鞋",
            "工作场所吸烟的",
            "切割作业时，附近的工作人员必须戴口罩，切割的操作人员必须戴面具、手套",
            "作业人员系挂了保险带，但是系挂不规范，有松动",
            "运料车、运人车门未关闭就行驶的"
        ]
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        logger.info(f"📋 已加载 {len(self.default_rules)} 条默认安全规则")
        logger.info("✅ 推理器初始化完成")
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"📥 加载模型: {model_path}")
        
        # 创建模型
        model = DynamicSafetyClassifier(preserve_aspect_ratio=True)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info("✅ 模型加载完成")
        return model
    
    def add_rules(self, new_rules):
        """添加新的安全规则"""
        if isinstance(new_rules, str):
            new_rules = [new_rules]
        
        original_count = len(self.default_rules)
        self.default_rules.extend(new_rules)
        
        logger.info(f"📋 添加 {len(new_rules)} 条新规则，总规则数: {len(self.default_rules)}")
        return len(self.default_rules) - original_count
    
    def predict_image(self, image_path, custom_rules=None, top_k=5):
        """
        对单张图片预测所有规则的违规概率
        
        Args:
            image_path: 图片路径
            custom_rules: 自定义规则列表（可选）
            top_k: 返回概率最高的前k条规则
        
        Returns:
            dict: 包含预测结果的字典
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 使用自定义规则或默认规则
        rules_to_test = custom_rules if custom_rules is not None else self.default_rules
        
        logger.info(f"🖼️ 分析图片: {image_path}")
        logger.info(f"📋 测试 {len(rules_to_test)} 条规则")
        
        # 加载和预处理图像
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).to(self.device)
        except Exception as e:
            raise ValueError(f"图片加载失败: {e}")
        
        # 对所有规则进行预测
        all_predictions = []
        
        # 批量处理以提高效率
        batch_size = 8  # 可以根据显存大小调整
        
        for i in range(0, len(rules_to_test), batch_size):
            batch_rules = rules_to_test[i:i+batch_size]
            batch_images = image_tensor.unsqueeze(0).repeat(len(batch_rules), 1, 1, 1)
            
            with torch.no_grad():
                batch_probabilities = self.model(batch_images, batch_rules)
                
            for j, prob in enumerate(batch_probabilities):
                rule_idx = i + j
                all_predictions.append({
                    'rule_idx': rule_idx,
                    'rule_text': batch_rules[j],
                    'probability': prob.item()
                })
        
        # 按概率排序
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # 获取概率最高的top_k条规则
        top_predictions = all_predictions[:top_k]
        
        # 构建结果
        result = {
            'image_path': image_path,
            'total_rules_tested': len(rules_to_test),
            'top_violations': top_predictions,
            'all_predictions': all_predictions
        }
        
        # 打印结果
        self._print_results(result)
        
        return result
    
    def _print_results(self, result):
        """打印预测结果"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 安全规则违规检测结果")
        logger.info(f"{'='*60}")
        logger.info(f"🖼️ 图片: {os.path.basename(result['image_path'])}")
        logger.info(f"📋 总测试规则: {result['total_rules_tested']} 条")
        logger.info(f"\n🔍 概率最高的 {len(result['top_violations'])} 条规则:")
        
        for i, pred in enumerate(result['top_violations'], 1):
            probability = pred['probability']
            status = "⚠️ 高风险" if probability > 0.7 else "⚡ 中风险" if probability > 0.3 else "✅ 低风险"
            
            logger.info(f"  {i}. [{probability:.4f}] {status}")
            logger.info(f"     📝 {pred['rule_text']}")
            logger.info("")
        
        # 统计违规情况
        high_risk = sum(1 for p in result['all_predictions'] if p['probability'] > 0.7)
        medium_risk = sum(1 for p in result['all_predictions'] if 0.3 < p['probability'] <= 0.7)
        low_risk = sum(1 for p in result['all_predictions'] if p['probability'] <= 0.3)
        
        logger.info(f"📈 风险统计:")
        logger.info(f"  ├── ⚠️ 高风险 (>0.7): {high_risk} 条")
        logger.info(f"  ├── ⚡ 中风险 (0.3-0.7): {medium_risk} 条")
        logger.info(f"  └── ✅ 低风险 (≤0.3): {low_risk} 条")

# ================================ 缩放测试 ================================

def run_scaling_performance_test(args):
    """运行规则缩放性能测试"""
    
    MODEL_PATH = args.model_path
    TEST_IMAGE = args.image_path
    RULE_COUNTS = args.rule_counts
    NUM_TESTS = 5
    
    logger.info("🚀 开始规则缩放性能测试")
    logger.info(f"📋 测试规则数量: {RULE_COUNTS}")
    logger.info(f"🖼️ 测试图片: {TEST_IMAGE}")
    logger.info(f"🔄 每个配置重复测试: {NUM_TESTS} 次")
    
    # 创建推理器
    inference = SafetyRuleInference(model_path=MODEL_PATH, device=args.device)
    
    # 扩展规则库
    extended_rules = inference.default_rules.copy()
    additional_rules = [
        "机械设备维修时未断电上锁", "焊接作业时未穿防护服装", "化学品储存区域无标识牌",
        "电气设备外壳未接地", "压力容器未按期检验", "消防器材被遮挡或移位",
        "作业现场无安全警示标志", "特种设备操作人员无证上岗", "危险品运输车辆超速行驶",
        "高温作业区域无降温措施", "有毒气体作业区域无通风设施", "密闭空间作业未进行气体检测",
        "起重作业时吊物下方站人", "移动脚手架作业时有人员在上方", "电焊作业现场堆放易燃物品",
        "高压设备停电后未验电就作业", "在运转的传送带上进行清理作业", "叉车行驶时货叉升得过高",
        "吊装作业中使用破损的吊具", "在强风天气进行高处作业", "未经许可进入受限空间",
        "使用损坏的个人防护设备", "在易燃易爆场所使用非防爆工具", "交叉作业时未设置安全隔离",
        "临时用电线路私拉乱接", "在禁止区域停放车辆", "安全标识缺失或不清晰",
        "应急设备被占用或损坏", "作业人员酒后上岗", "未按规定穿戴劳防用品",
        "违章指挥或违章作业", "安全培训记录不完整", "隐患排查整改不及时",
        "安全责任制落实不到位", "应急预案演练不足", "安全投入不能满足需要",
        "职业健康监护不规范", "危险源识别评价不全面", "安全文化建设薄弱",
        "安全绩效考核不严格", "作业环境风险评估不充分", "个人防护设备配置不当",
        "安全操作规程执行不严", "应急救援装备维护不到位", "职业病防护措施不完善",
        "危险化学品存储不规范", "特殊作业审批程序不完整", "安全监督检查频次不够",
        "事故隐患治理不彻底", "安全教育培训针对性不强"
    ]
    
    extended_rules.extend(additional_rules)
    
    # 确保有足够的规则
    max_rules_needed = max(RULE_COUNTS)
    while len(extended_rules) < max_rules_needed:
        base_rule = extended_rules[len(extended_rules) % len(inference.default_rules)]
        variant_rule = f"{base_rule}（变体{len(extended_rules) - len(inference.default_rules) + 1}）"
        extended_rules.append(variant_rule)
    
    logger.info(f"📋 扩展规则库至 {len(extended_rules)} 条规则")
    
    # 存储测试结果
    all_results = []
    
    for rule_count in RULE_COUNTS:
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 测试 {rule_count} 条规则")
        logger.info(f"{'='*50}")
        
        test_rules = extended_rules[:rule_count]
        inference_times = []
        
        for test_idx in range(NUM_TESTS):
            logger.info(f"  第 {test_idx + 1}/{NUM_TESTS} 次测试...")
            
            start_time = time.perf_counter()
            
            # 禁用打印输出
            original_print_results = inference._print_results
            inference._print_results = lambda x: None
            
            try:
                result = inference.predict_image(
                    image_path=TEST_IMAGE,
                    custom_rules=test_rules,
                    top_k=5
                )
            finally:
                inference._print_results = original_print_results
            
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            logger.info(f"    ⏱️ 推理时间: {inference_time:.3f}s")
        
        # 计算统计数据
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        result_stats = {
            'num_rules': rule_count,
            'inference_times': inference_times,
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time
        }
        
        all_results.append(result_stats)
        
        logger.info(f"📊 统计结果: {mean_time:.3f}±{std_time:.3f}s")
        logger.info(f"   范围: {min_time:.3f}s - {max_time:.3f}s")
    
    # 保存和可视化结果
    output_dir = '/data/LLM/WSC/moe/rule_scaling_results'
    os.makedirs(output_dir, exist_ok=True)
    
    save_results(all_results, RULE_COUNTS, NUM_TESTS, TEST_IMAGE, MODEL_PATH, args.device, output_dir)
    generate_visualization(all_results, output_dir)
    print_results_table(all_results)
    
    return all_results

def save_results(results, rule_counts, num_tests, test_image, model_path, device, output_dir):
    """保存测试结果"""
    results_file = os.path.join(output_dir, 'scaling_test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_config': {
                'rule_counts': rule_counts,
                'num_tests': num_tests,
                'test_image': test_image,
                'model_path': model_path,
                'device': device,
                'test_time': datetime.now().isoformat()
            },
            'results': results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"📁 详细结果已保存至: {results_file}")

def generate_visualization(results, output_dir):
    """生成可视化图表"""
    logger.info("🎨 生成缩放性能图表...")
    
    rule_counts = [r['num_rules'] for r in results]
    mean_times = [r['mean_time'] for r in results]
    std_times = [r['std_time'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#8e44ad', '#3498db', '#16a085', '#27ae60', '#f39c12']
    
    bars = ax.bar(range(len(rule_counts)), mean_times, 
                 yerr=std_times, capsize=8, capthick=2,
                 color=colors, alpha=0.85, 
                 edgecolor='black', linewidth=1.5)
    
    for i, (bar, mean_time, std_time) in enumerate(zip(bars, mean_times, std_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
               height + std_time + max(mean_times) * 0.02,
               f'{mean_time:.3f}s',
               ha='center', va='bottom', 
               fontsize=12, fontweight='bold', color='black')
    
    ax.set_xlabel('Number of Rules', fontsize=16, fontweight='bold')
    ax.set_ylabel('Inference Time (seconds)', fontsize=16, fontweight='bold')
    
    ax.set_xticks(range(len(rule_counts)))
    ax.set_xticklabels(rule_counts, fontsize=14)
    
    y_max = max([m + s for m, s in zip(mean_times, std_times)]) * 1.15
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='y', labelsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('gray')
    
    plt.tight_layout()
    
    chart_path_png = os.path.join(output_dir, 'rule_scaling_performance.png')
    chart_path_pdf = os.path.join(output_dir, 'rule_scaling_performance.pdf')
    
    plt.savefig(chart_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(chart_path_pdf, bbox_inches='tight', facecolor='white')
    
    logger.info(f"📊 图表已保存至:")
    logger.info(f"   PNG: {chart_path_png}")
    logger.info(f"   PDF: {chart_path_pdf}")
    
    plt.show()

def print_results_table(results):
    """打印结果汇总表格"""
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 Rule Scaling Performance Test Results")
    logger.info(f"{'='*80}")
    
    print(f"{'Rules':>6} | {'Mean(s)':>8} | {'Std(s)':>7} | {'Min(s)':>7} | {'Max(s)':>7} | {'vs Base':>8} | {'Efficiency':>10}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}")
    
    base_time = results[0]['mean_time']
    
    for result in results:
        rules = result['num_rules']
        mean_t = result['mean_time']
        std_t = result['std_time']
        min_t = result['min_time']
        max_t = result['max_time']
        vs_base = mean_t / base_time
        efficiency = (1 - 5/rules) * 100
        
        print(f"{rules:6d} | {mean_t:8.3f} | {std_t:7.3f} | {min_t:7.3f} | {max_t:7.3f} | {vs_base:8.2f}x | {efficiency:9.1f}%")
    
    logger.info(f"{'='*80}")

# ================================ 主函数 ================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='安全规则违规检测推理')
    parser.add_argument('--model_path', 
                       default='/data/LLM/wjj/moe/net_save_enhanced/best_dynamic_model.pth',
                       help='模型文件路径')
    parser.add_argument('--image_path', 
                        default='/data/LLM/WSC/dataset_video/extracted_frames/43-5.jpg',
                       help='要检测的图片路径')
    parser.add_argument('--top_k', 
                       type=int, 
                       default=5,
                       help='返回概率最高的前k条规则')
    parser.add_argument('--device', 
                       default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备选择')
    parser.add_argument('--scaling_test', 
                       action='store_true',
                       help='运行规则数量缩放性能测试')
    parser.add_argument('--rule_counts',
                       nargs='+',
                       type=int,
                       default=[27, 37, 47, 57, 67],
                       help='缩放测试的规则数量列表')
    
    args = parser.parse_args()
    
    try:
        if args.scaling_test:
            logger.info("🚀 启动规则数量缩放性能测试模式")
            run_scaling_performance_test(args)
        else:
            # 常规推理
            inference = SafetyRuleInference(
                model_path=args.model_path,
                device=args.device
            )
            
            new_rules = ["机械设备维修时未断电上锁"]
            inference.add_rules(new_rules)
            
            start_time = datetime.now()
            result = inference.predict_image(
                image_path=args.image_path,
                top_k=args.top_k
            )
            inference_time = (datetime.now() - start_time).total_seconds()
            print(f"🔍 推理完成，耗时: {inference_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"❌ 推理失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()