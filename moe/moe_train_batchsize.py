#!/usr/bin/env python3
"""
动态规则安全违规检测模型训练脚本 - 支持任意数量规则的单规则-图片对输入
模型保存到固定路径: /home/baoadmin/LLM/WSC/moe/net_save2
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================ 固定保存路径配置 ================================
MODEL_SAVE_DIR = "/home/baoadmin/LLM/WSC/moe/net_save_enhanced"

def ensure_save_directory():
    """确保保存目录存在"""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        logger.info(f"📁 创建保存目录: {MODEL_SAVE_DIR}")
    else:
        logger.info(f"📁 使用保存目录: {MODEL_SAVE_DIR}")

def get_model_save_path(prefix="dynamic_safety_model"):
    """生成模型保存路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pth"
    return os.path.join(MODEL_SAVE_DIR, filename)

def get_best_model_path():
    """获取最佳模型的固定路径"""
    return os.path.join(MODEL_SAVE_DIR, "best_dynamic_model.pth")

def get_final_model_path():
    """获取最终模型的固定路径"""
    return os.path.join(MODEL_SAVE_DIR, "final_dynamic_model.pth")

# ================================ 动态规则数据集 ================================

def custom_collate_fn(batch):
    """自定义collate函数 - 处理不同尺寸的图像和规则文本"""
    images = []
    rule_texts = []
    targets = []
    image_paths = []
    
    # 找到batch中的最大图像尺寸
    max_h = max([item['image'].shape[1] for item in batch])
    max_w = max([item['image'].shape[2] for item in batch])
    
    for item in batch:
        img = item['image']  # [3, H, W]
        
        # 计算padding
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        # 应用padding (左,右,上,下)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        images.append(padded_img)
        rule_texts.append(item['rule_text'])
        targets.append(item['target'])
        image_paths.append(item['image_path'])
    
    return {
        'image': torch.stack(images),
        'rule_text': rule_texts,  # 保持为字符串列表
        'target': torch.stack(targets),
        'image_paths': image_paths
    }

class DynamicSafetyDataset(Dataset):
    """动态规则安全数据集 - 每个样本是单个规则-图片对"""
    
    def __init__(self, json_file, rules_list=None, transform=None, preserve_aspect_ratio=True):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 默认规则列表（可以从外部传入更多规则）
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
            "皮带机不停机，工作人员在它附近进行工作的",
            "起重机械启动过程中，人员上下车或人员靠的起重机械很近",
            "在临边、洞口处工作不系安全带",
            "在专业的实验室里面，工作人员没穿劳保鞋",
            "工作场所吸烟的",
            "切割作业时，附近的工作人员必须戴口罩，切割的操作人员必须戴面具、手套",
            "作业人员系挂了保险带，但是系挂不规范，有松动",
            "运料车、运人车门未关闭就行驶的",
            "擅自启动已停机挂牌设备",
            "配电箱（柜）内或上部摆放其他物品",
            "车辆运行中，未系安全带的",
            "搬运氧、乙炔气瓶时在地面上滚动搬运",
            "开车挤、撞坏溜井栏杆的",
            "人员站在举升的铲斗内检修、施工、登高及从事其他作业的",
            "人员站在矿车前方拉矿车",
            "上下楼梯未抓扶手"
        ]
        
        # 使用传入的规则列表或默认规则列表
        self.rules_list = rules_list if rules_list is not None else self.default_rules
        logger.info(f"📋 使用 {len(self.rules_list)} 条安全规则")
        
        # 图像变换
        if preserve_aspect_ratio:
            self.transform = transform or transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("✅ 使用原始图片尺寸（保持宽高比）")
        else:
            input_size = 224
            self.transform = transform or transforms.Compose([
                transforms.Resize(int(input_size * 1.15)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info(f"✅ 使用固定尺寸: {input_size}×{input_size}")
        
        # 生成规则-图片对
        self.rule_image_pairs = self._generate_pairs()
        logger.info(f"📂 生成 {len(self.rule_image_pairs)} 个规则-图片对")
        
    def _generate_pairs(self):
        """生成所有规则-图片对"""
        pairs = []
        
        for sample in self.data:
            image_path = sample['image']
            
            # 获取该图片的违规规则信息
            rule_encoding = sample.get('rule_encoding', [0] * len(self.rules_list))
            # 确保长度匹配
            if len(rule_encoding) < len(self.rules_list):
                rule_encoding.extend([0] * (len(self.rules_list) - len(rule_encoding)))
            elif len(rule_encoding) > len(self.rules_list):
                rule_encoding = rule_encoding[:len(self.rules_list)]
            
            # 为每条规则创建一个训练样本
            for rule_idx, rule_text in enumerate(self.rules_list):
                target = float(rule_encoding[rule_idx])  # 0 或 1
                
                pairs.append({
                    'image_path': image_path,
                    'rule_text': rule_text,
                    'rule_idx': rule_idx,
                    'target': target,
                    'original_sample': sample
                })
        
        return pairs
    
    def __len__(self):
        return len(self.rule_image_pairs)
    
    def __getitem__(self, idx):
        pair = self.rule_image_pairs[idx]
        
        # 加载图像
        img = Image.open(pair['image_path']).convert('RGB')
        image_input = self.transform(img)
        
        # 获取规则文本和目标
        rule_text = pair['rule_text']
        target = torch.tensor(pair['target'], dtype=torch.float32)
        
        return {
            'image': image_input,           # [3, H, W]
            'rule_text': rule_text,         # str
            'target': target,               # scalar (0.0 or 1.0)
            'image_path': pair['image_path'],
            'rule_idx': pair['rule_idx']
        }
    
    def add_rules(self, new_rules):
        """动态添加新规则"""
        original_count = len(self.rules_list)
        self.rules_list.extend(new_rules)
        logger.info(f"📋 添加 {len(new_rules)} 条新规则，总规则数: {len(self.rules_list)}")
        
        # 重新生成规则-图片对（新规则的标签默认为0）
        self.rule_image_pairs = self._generate_pairs()
        logger.info(f"📂 重新生成 {len(self.rule_image_pairs)} 个规则-图片对")

class DynamicSafetyClassifier(nn.Module):
    """动态规则安全分类器 - 输入单条规则和图片，输出单个概率"""
    
    def __init__(self, hidden_dim=1024, preserve_aspect_ratio=True):
        super().__init__()
        
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # 1. 图像编码器 (ResNet50) - 冻结
        logger.info("🔧 加载ResNet50并冻结参数...")
        self.image_encoder = models.resnet50(weights='IMAGENET1K_V1')
        
        if preserve_aspect_ratio:
            self.image_encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.image_encoder.fc = nn.Identity()
            logger.info("✅ ResNet50配置为支持任意尺寸输入，输出: [B, 2048]")
        else:
            self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
            logger.info("✅ ResNet50配置为固定尺寸输入(224x224)，输出维度: 2048")
        
        # 冻结ResNet50的所有参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()
        
        # 2. 文本编码器 (Chinese Longformer) - 冻结
        logger.info("🔧 加载Chinese Longformer并冻结参数...")
        
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
        
        # 3. 多模态融合分类器 (只有这部分参与训练)
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
            
            nn.Linear(hidden_dim // 4, 1),             # 256 -> 1 (单个概率)
            nn.Sigmoid()  # 输出0-1概率
        )
        
        self._print_model_info()
        
    def _print_model_info(self):
        """打印模型参数信息"""
        resnet_params = sum(p.numel() for p in self.image_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        text_dim = self.text_encoder.config.hidden_size
        
        logger.info("📊 模型参数统计:")
        logger.info(f"  ├── ResNet50: {resnet_params:,} 参数 (冻结)")
        logger.info(f"  ├── Text Encoder: {text_params:,} 参数 (冻结)")  
        logger.info(f"  │   └── 输出维度: {text_dim}")
        logger.info(f"  ├── 分类器: {classifier_params:,} 参数 (可训练)")
        logger.info(f"  ├── 总参数: {total_params:,}")
        logger.info(f"  └── 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
    def encode_rule_texts(self, rule_texts):
        """编码规则文本（batch处理）"""
        with torch.no_grad():
            # 批量编码文本
            encoded = self.tokenizer(
                rule_texts,
                truncation=True,
                padding=True,
                max_length=512,  # 单条规则用512长度足够
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
            predictions: [B, 1] 违规概率
        """
        batch_size = images.size(0)
        
        # 1. 图像特征提取 (冻结的ResNet50)
        with torch.no_grad():
            image_features = self.image_encoder(images)  # [B, 2048]
            image_features = image_features.view(batch_size, -1)
        
        # 2. 文本特征提取 (冻结的文本编码器)
        text_features = self.encode_rule_texts(rule_texts)  # [B, text_dim]
        
        # 3. 特征融合
        combined_features = torch.cat([image_features, text_features], dim=1)  # [B, 2048+text_dim]
        
        # 4. 分类预测 (只有这部分参与训练)
        predictions = self.classifier(combined_features)  # [B, 1]
        
        return predictions.squeeze(-1)  # [B] 返回标量概率

class DynamicSafetyTrainer:
    """动态规则安全模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, lr=1e-3, loss_type='bce'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_type = loss_type
        
        # 只优化分类器参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 损失函数选择
        if loss_type == 'bce':
            self.criterion = nn.BCELoss()
            logger.info("📊 使用BCE Loss (二元交叉熵)")
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
            logger.info("📊 使用MSE Loss (均方误差)")
        elif loss_type == 'focal':
            self.criterion = self._focal_loss
            logger.info("📊 使用Focal Loss (处理类别不平衡)")
        else:
            raise ValueError(f"不支持的loss类型: {loss_type}")
        logger.info(f"🎯 可训练参数: {sum(p.numel() for p in trainable_params):,}")
        
    def _focal_loss(self, pred, target, alpha=1, gamma=2):
        """Focal Loss实现"""
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
        
    def train_epoch(self):
        self.model.train()
        # 冻结的部分保持eval模式
        self.model.image_encoder.eval()
        self.model.text_encoder.eval()
        
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="训练")):
            images = batch['image'].to(self.device)
            rule_texts = batch['rule_text']  # 文本列表
            targets = batch['target'].to(self.device)  # [B]
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(images, rule_texts)  # [B]
            
            # 计算损失
            loss = self.criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 每100个batch打印一次详细信息
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                images = batch['image'].to(self.device)
                rule_texts = batch['rule_text']
                targets = batch['target'].to(self.device)
                
                predictions = self.model(images, rule_texts)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算评估指标
        all_preds = torch.cat(all_preds, dim=0)  # [N]
        all_targets = torch.cat(all_targets, dim=0)  # [N]
        
        metrics = self._compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _compute_metrics(self, predictions, targets, threshold=0.5):
        """计算评估指标"""
        pred_binary = (predictions > threshold).float()
        target_binary = (targets > threshold).float()
        
        # 准确率
        accuracy = (pred_binary == target_binary).float().mean().item()
        
        # 精确率、召回率、F1
        tp = ((pred_binary == 1) & (target_binary == 1)).sum().item()
        fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()
        tn = ((pred_binary == 0) & (target_binary == 0)).sum().item()
        fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    def train(self, num_epochs=50, save_best=True, save_checkpoints=True, save_final=True):
        """训练模型"""
        
        best_model_path = get_best_model_path() if save_best else None
        final_model_path = get_final_model_path() if save_final else None
        
        best_val_loss = float('inf')
        best_f1 = 0.0
        
        logger.info(f"🚀 开始训练 {num_epochs} 个epochs")
        logger.info(f"💾 模型保存目录: {MODEL_SAVE_DIR}")
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, metrics = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            logger.info(f"📊 训练损失: {train_loss:.4f}")
            logger.info(f"📊 验证损失: {val_loss:.4f}")
            logger.info(f"📊 准确率: {metrics['accuracy']:.4f}")
            logger.info(f"📊 精确率: {metrics['precision']:.4f}")
            logger.info(f"📊 召回率: {metrics['recall']:.4f}")
            logger.info(f"📊 F1分数: {metrics['f1']:.4f}")
            
            # 保存最佳模型
            is_best = False
            if val_loss < best_val_loss or metrics['f1'] > best_f1:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                is_best = True
                
                if save_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'best_f1': best_f1,
                        'metrics': metrics,
                        'loss_type': self.loss_type,
                        'model_config': {
                            'preserve_aspect_ratio': self.model.preserve_aspect_ratio,
                        }
                    }, best_model_path)
                    logger.info(f"✅ 保存最佳模型 (Val Loss: {val_loss:.4f}, F1: {metrics['f1']:.4f})")
            
            # 保存检查点
            if save_checkpoints and (epoch + 1) % 10 == 0:
                checkpoint_path = get_model_save_path(f"checkpoint_epoch_{epoch+1}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'loss_type': self.loss_type
                }, checkpoint_path)
                logger.info(f"💾 保存检查点: {checkpoint_path}")
        
        # 保存最终模型
        if save_final:
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'best_f1': best_f1,
                'final_metrics': metrics,
                'loss_type': self.loss_type,
                'model_config': {
                    'preserve_aspect_ratio': self.model.preserve_aspect_ratio,
                }
            }, final_model_path)
            logger.info(f"💾 保存最终模型: {final_model_path}")
        
        logger.info(f"\n🎉 训练完成!")
        logger.info(f"📊 最佳验证损失: {best_val_loss:.4f}")
        logger.info(f"📊 最佳F1分数: {best_f1:.4f}")

def main():
    """主函数"""
    ensure_save_directory()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 使用设备: {device}")
    
    # 数据集路径
    dataset_path = '/data/LLM/WSC/moe/v2/safety_analysis_moe_v2.json'
    
    # 配置选项
    PRESERVE_ASPECT_RATIO = True
    
    # 创建动态数据集
    dataset = DynamicSafetyDataset(dataset_path, preserve_aspect_ratio=PRESERVE_ASPECT_RATIO)
    
    # 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # 数据加载器
    batch_size = 16 if PRESERVE_ASPECT_RATIO else 32
    collate_fn = custom_collate_fn if PRESERVE_ASPECT_RATIO else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )
    
    logger.info(f"📊 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    logger.info(f"📊 批次大小: {batch_size}")
    logger.info(f"📊 总规则-图片对: {len(dataset)}")
    
    # 创建动态模型
    model = DynamicSafetyClassifier(hidden_dim=1024, preserve_aspect_ratio=PRESERVE_ASPECT_RATIO)
    
    logger.info(f"\n🏗️ 动态规则模型架构:")
    logger.info("  ├── 图像编码器: ResNet50 → 2048维 (冻结)")
    logger.info("  ├── 文本编码器: Chinese Longformer → 768维 (冻结)")
    logger.info("  ├── 特征融合: 2048 + 768 = 2816维")
    logger.info("  └── 分类器: 2816 → 1024 → 512 → 256 → 1 (可训练)")
    logger.info("  📝 输入: 单条规则文本 + 图片")
    logger.info("  📝 输出: 单个违规概率 [0-1]")
    
    # 创建训练器
    trainer = DynamicSafetyTrainer(
        model, train_loader, val_loader, device, 
        lr=1e-3, 
        loss_type='bce'
    )
    
    # 开始训练
    trainer.train(
        num_epochs=50, 
        save_best=True,
        save_checkpoints=True, 
        save_final=True
    )

# ================================ 推理示例函数 ================================

def load_model_for_inference(model_path, device='cpu'):
    """加载训练好的模型用于推理"""
    # 创建模型
    model = DynamicSafetyClassifier(preserve_aspect_ratio=True)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✅ 模型加载完成: {model_path}")
    return model

def predict_single_rule_image(model, image_path, rule_text, device='cpu'):
    """
    对单个规则-图片对进行预测
    
    Args:
        model: 训练好的模型
        image_path: 图片路径
        rule_text: 规则文本
        device: 设备
    
    Returns:
        float: 违规概率 [0-1]
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # 准备规则文本
    rule_texts = [rule_text]  # 批次为1
    
    # 预测
    with torch.no_grad():
        probability = model(image_tensor, rule_texts)  # [1]
        probability = probability.item()
    
    return probability

def predict_multiple_rules(model, image_path, rules_list, device='cpu', threshold=0.5):
    """
    对单张图片和多条规则进行预测
    
    Args:
        model: 训练好的模型
        image_path: 图片路径
        rules_list: 规则文本列表
        device: 设备
        threshold: 违规阈值
    
    Returns:
        dict: 包含所有规则的预测结果
    """
    results = {}
    violated_rules = []
    
    for i, rule_text in enumerate(rules_list):
        probability = predict_single_rule_image(model, image_path, rule_text, device)
        
        results[f"rule_{i}"] = {
            'rule_text': rule_text,
            'probability': probability,
            'violated': probability > threshold
        }
        
        if probability > threshold:
            violated_rules.append({
                'rule_idx': i,
                'rule_text': rule_text,
                'probability': probability
            })
    
    return {
        'image_path': image_path,
        'all_predictions': results,
        'violated_rules': violated_rules,
        'total_violations': len(violated_rules)
    }

# ================================ 使用示例 ================================

def example_usage():
    """使用示例"""
    
    # 1. 训练模型
    logger.info("🚀 开始训练动态规则模型...")
    main()
    
    # 2. 加载训练好的模型进行推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = get_best_model_path()
    model = load_model_for_inference(model_path, device)
    
    # 3. 单规则预测示例
    image_path = "/path/to/test/image.jpg"
    rule_text = "生产作业场所未佩戴安全帽的"
    
    probability = predict_single_rule_image(model, image_path, rule_text, device)
    logger.info(f"📝 规则: {rule_text}")
    logger.info(f"🖼️ 图片: {image_path}")
    logger.info(f"📊 违规概率: {probability:.4f}")
    
    # 4. 多规则预测示例
    rules_list = [
        "生产作业场所未佩戴安全帽的",
        "工作场所吸烟的",
        "使用角磨机未佩戴防护眼镜",
        "叉车载人的"
    ]
    
    results = predict_multiple_rules(model, image_path, rules_list, device)
    logger.info(f"\n📊 多规则预测结果:")
    logger.info(f"📂 图片: {results['image_path']}")
    logger.info(f"⚠️ 违规规则数: {results['total_violations']}")
    
    for violation in results['violated_rules']:
        logger.info(f"  ├── 规则 {violation['rule_idx']}: {violation['rule_text']}")
        logger.info(f"  └── 概率: {violation['probability']:.4f}")

if __name__ == "__main__":
    main()