import json
import re
import numpy as np
from typing import Dict, List, Tuple

class MoEDatasetGenerator:
    def __init__(self):
        self.all_rules = [
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
        
        # 创建规则到索引的映射
        self.rule_to_idx = {rule: idx for idx, rule in enumerate(self.all_rules)}
    
    def extract_relevant_rules(self, assistant_content: str) -> List[str]:
        """从助手回复中提取相关规则"""
        relevant_rules = []
        
        print(f"开始提取相关规则...")
        
        # 定义多个可能的相关规则识别模式
        patterns = [
            r"相关规则识别[：:]\s*(.*?)(?=\n#### |##### |### |\Z)",
            r"适用的安全规则[包括：:]*\s*(.*?)(?=\n#### |##### |### |\Z)",
            r"以下安全规则可能适用[：:]\s*(.*?)(?=\n#### |##### |### |\Z)"
        ]
        
        content_found = None
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, assistant_content, re.DOTALL)
            if matches:
                content_found = matches[0]
                print(f"找到相关规则部分 (模式{i+1}): {content_found[:200]}...")
                break
        
        if content_found:
            # 修复的正则表达式 - 处理数字编号的规则 (1. 2. 3.)
            number_pattern = r"(\d+)\.\s*([^\n]+)"
            number_matches = re.findall(number_pattern, content_found)
            
            for number, rule_text in number_matches:
                rule_text = rule_text.strip()
                
                # 保存原始文本用于调试
                original_rule_text = rule_text
                
                # 去除括号中的说明文字
                rule_text = re.sub(r'（.*?）', '', rule_text)
                rule_text = re.sub(r'\(.*?\)', '', rule_text)
                rule_text = rule_text.strip()
                
                print(f"正在匹配规则文本: '{original_rule_text}' -> '{rule_text}'")
                
                # 尝试完整匹配
                if rule_text in self.all_rules:
                    relevant_rules.append(rule_text)
                    print(f"✓ 完整匹配规则: {rule_text}")
                    continue
                
                # 尝试去掉结尾的"的"字再匹配
                rule_text_no_de = rule_text.rstrip('的')
                for full_rule in self.all_rules:
                    if full_rule == rule_text_no_de or full_rule == rule_text_no_de + '的':
                        relevant_rules.append(full_rule)
                        print(f"✓ 去除/添加'的'字后匹配规则: {full_rule}")
                        break
                else:
                    # 尝试模糊匹配
                    matched_rule = self._fuzzy_match_rule(rule_text)
                    if matched_rule:
                        relevant_rules.append(matched_rule)
                        print(f"✓ 模糊匹配规则: {rule_text} -> {matched_rule}")
                    else:
                        print(f"✗ 未匹配到规则: {rule_text}")
            
            # 也处理可能的 - 或 * 开头的规则
            lines = content_found.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    rule_text = line[2:].strip()
                    # 去除括号中的说明文字
                    rule_text = re.sub(r'（.*?）', '', rule_text)
                    rule_text = re.sub(r'\(.*?\)', '', rule_text)
                    rule_text = rule_text.strip()
                    
                    print(f"正在匹配规则文本: '{rule_text}'")
                    
                    # 尝试完整匹配
                    if rule_text in self.all_rules:
                        relevant_rules.append(rule_text)
                        print(f"✓ 完整匹配规则: {rule_text}")
                        continue
                    
                    # 尝试模糊匹配
                    matched_rule = self._fuzzy_match_rule(rule_text)
                    if matched_rule:
                        relevant_rules.append(matched_rule)
                        print(f"✓ 模糊匹配规则: {rule_text} -> {matched_rule}")
                    else:
                        print(f"✗ 未匹配到规则: {rule_text}")
        else:
            print("未找到任何相关规则识别部分")
        
        # 从"逐项检查"部分提取规则
        check_patterns = [
            r"逐项检查[：:]\s*(.*?)(?=#### |##### |### |\Z)"
        ]
        
        check_content_found = None
        for pattern in check_patterns:
            matches = re.findall(pattern, assistant_content, re.DOTALL)
            if matches:
                check_content_found = matches[0]
                print(f"找到逐项检查部分: {check_content_found[:300]}...")
                break
        
        if check_content_found:
            # 提取形如 "1. **规则名称**" 的内容
            rule_pattern = r"\d+\.\s*\*\*(.*?)\*\*"
            rule_matches = re.findall(rule_pattern, check_content_found)
            print(f"从逐项检查中提取到的规则文本: {rule_matches}")
            
            for rule_text in rule_matches:
                rule_text = rule_text.strip()
                print(f"正在匹配规则文本: '{rule_text}'")
                
                # 尝试完整匹配
                if rule_text in self.all_rules:
                    relevant_rules.append(rule_text)
                    print(f"✓ 从逐项检查中完整匹配规则: {rule_text}")
                    continue
                
                # 尝试模糊匹配
                matched_rule = self._fuzzy_match_rule(rule_text)
                if matched_rule:
                    relevant_rules.append(matched_rule)
                    print(f"✓ 从逐项检查中模糊匹配规则: {rule_text} -> {matched_rule}")
                else:
                    print(f"✗ 从逐项检查中未匹配到规则: {rule_text}")
        else:
            print("未找到逐项检查部分")
        
        # 去重
        relevant_rules = list(set(relevant_rules))
        print(f"最终提取到的相关规则: {relevant_rules}")
        
        return relevant_rules
    
    def _fuzzy_match_rule(self, text_rule: str) -> str:
        """改进的模糊匹配方法"""
        best_match = None
        best_score = 0
        
        print(f"  模糊匹配输入: '{text_rule}'")
        
        # 提取文本规则的关键词
        text_keywords = set(re.findall(r'[\u4e00-\u9fff]+', text_rule))
        print(f"  提取的关键词: {text_keywords}")
        
        for i, full_rule in enumerate(self.all_rules):
            # 提取完整规则的关键词
            full_keywords = set(re.findall(r'[\u4e00-\u9fff]+', full_rule))
            
            if len(text_keywords) > 0:
                # 计算关键词重叠度 - 使用更宽松的计算方式
                overlap = len(text_keywords & full_keywords)
                
                # 改进相似度计算：考虑双向匹配
                similarity1 = overlap / len(text_keywords)  # 输入文本的覆盖度
                similarity2 = overlap / len(full_keywords)  # 规则库文本的覆盖度
                similarity = max(similarity1, similarity2)  # 取较高的相似度
                
                # 特殊处理：如果有很高的关键词重叠，降低阈值
                if overlap >= 3:  # 如果有3个或以上关键词匹配
                    min_threshold = 0.2
                else:
                    min_threshold = 0.3
                
                # 打印匹配详情
                if overlap > 0:
                    print(f"  规则{i}: 重叠{overlap} 相似度{similarity:.2f} - {full_rule}")
                
                # 如果相似度大于阈值且是当前最佳匹配
                if similarity > min_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = full_rule
        
        if best_match:
            print(f"  最佳匹配 (得分{best_score:.2f}): {best_match}")
        else:
            print(f"  未找到匹配规则")
        
        return best_match
    
    def extract_violated_rules(self, assistant_content: str) -> Tuple[List[str], List[str]]:
        """提取确认违反和疑似违反的规则"""
        confirmed_violations = []
        suspected_violations = []
        
        print(f"开始提取违反规则...")
        
        # 查找违反规则部分 - 多种可能的格式
        violation_patterns = [
            r"### 违反的规则[：:]\s*(.*?)(?=\n### |$)",
            r"违反的规则[：:]\s*(.*?)(?=\n### |$)",
            r"### 总体结论[：:]\s*(.*?)(?=\n### |$)"
        ]
        
        violation_section = None
        for pattern in violation_patterns:
            match = re.search(pattern, assistant_content, re.DOTALL)
            if match:
                violation_section = match.group(1)
                print(f"找到违反规则部分: {violation_section[:200]}...")
                break
        
        if violation_section:
            # 提取确认违反
            confirmed_patterns = [
                r"\*\s*\*\*确认违反\*\*[：:]\s*(.*?)(?=\n\s*\*|\n\s*-|$)",
                r"确认违反[：:]\s*(.*?)(?=\n|$)"
            ]
            
            for pattern in confirmed_patterns:
                confirmed_match = re.search(pattern, violation_section, re.DOTALL)
                if confirmed_match:
                    confirmed_text = confirmed_match.group(1).strip()
                    print(f"确认违反文本: '{confirmed_text}'")
                    if "无" not in confirmed_text and confirmed_text:
                        # 按分号或中文句号分割
                        rules = re.split(r'[；;。]', confirmed_text)
                        for rule in rules:
                            rule = rule.strip()
                            if rule and rule != "无":
                                # 尝试匹配到完整规则
                                matched_rule = self._match_to_full_rule(rule)
                                if matched_rule:
                                    confirmed_violations.append(matched_rule)
                                    print(f"✓ 确认违反规则: {matched_rule}")
                    break
            
            # 提取疑似违反
            suspected_patterns = [
                r"\*\s*\*\*疑似违反\*\*[：:]\s*(.*?)(?=\n\s*\*|\n\s*-|$)",
                r"疑似违反[：:]\s*(.*?)(?=\n|$)"
            ]
            
            for pattern in suspected_patterns:
                suspected_match = re.search(pattern, violation_section, re.DOTALL)
                if suspected_match:
                    suspected_text = suspected_match.group(1).strip()
                    print(f"疑似违反文本: '{suspected_text}'")
                    if "无" not in suspected_text and suspected_text:
                        # 按分号或中文句号分割
                        rules = re.split(r'[；;。]', suspected_text)
                        for rule in rules:
                            rule = rule.strip()
                            if rule and rule != "无":
                                # 尝试匹配到完整规则
                                matched_rule = self._match_to_full_rule(rule)
                                if matched_rule:
                                    suspected_violations.append(matched_rule)
                                    print(f"✓ 疑似违反规则: {matched_rule}")
                    break
        else:
            print("未找到违反规则部分")
        
        print(f"最终提取到 - 确认违反: {confirmed_violations}, 疑似违反: {suspected_violations}")
        return confirmed_violations, suspected_violations
    
    def _match_to_full_rule(self, rule_text: str) -> str:
        """将文本中的规则匹配到完整规则"""
        # 尝试完整匹配
        if rule_text in self.all_rules:
            return rule_text
        
        # 尝试模糊匹配
        return self._fuzzy_match_rule(rule_text)
    
    def generate_rule_encoding(self, relevant_rules: List[str], 
                              confirmed_violations: List[str], 
                              suspected_violations: List[str]) -> List[int]:
        """生成规则的ONE-HOT编码，相关的为1，无关的为0"""
        one_hot_encoding = [0] * len(self.all_rules)
        
        # 所有相关规则（包括相关、确认违反、疑似违反）都标记为1
        all_relevant = set(relevant_rules + confirmed_violations + suspected_violations)
        
        for rule in all_relevant:
            if rule in self.rule_to_idx:
                one_hot_encoding[self.rule_to_idx[rule]] = 1
        
        return one_hot_encoding
    
    def process_json_file(self, input_file: str, output_file: str):
        """处理JSON文件并生成MoE训练数据集"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        moe_dataset = []
        
        # 处理不同的数据格式
        if isinstance(data, list):
            # 如果是列表格式
            items = data
        elif isinstance(data, dict):
            # 如果是字典格式，可能有其他结构
            items = [data]
        else:
            print(f"不支持的数据格式: {type(data)}")
            return
        
        for item in items:
            if isinstance(item, dict):
                # 提取图像路径 - 只取第一张图片
                images = None
                if 'images' in item:
                    images = item['images']
                elif 'image' in item:
                    images = [item['image']]  # 统一为列表格式处理
                
                if images and isinstance(images, list) and len(images) > 0:
                    # 只保存第一张图片
                    single_image = images[0]
                    print(f"处理图片: {single_image}")
                else:
                    print(f"跳过样本，没有找到有效图片: {item.keys()}")
                    continue  # 如果没有图片就跳过
                
                # 找到assistant的回复
                assistant_content = ""
                if 'messages' in item:
                    for msg in item['messages']:
                        if msg.get('role') == 'assistant':
                            assistant_content = msg.get('content', '')
                            break
                elif 'content' in item:
                    # 如果直接有content字段
                    assistant_content = item['content']
                
                if assistant_content:
                    print(f"\n=== 处理新样本 ===")
                    
                    # 提取相关规则
                    relevant_rules = self.extract_relevant_rules(assistant_content)
                    
                    # 提取违反的规则
                    confirmed_violations, suspected_violations = self.extract_violated_rules(assistant_content)
                    
                    # 生成ONE-HOT编码
                    rule_encoding = self.generate_rule_encoding(
                        relevant_rules, confirmed_violations, suspected_violations
                    )
                    
                    # 创建MoE训练样本 - 只包含一张图片
                    moe_sample = {
                        "image": single_image,  # 单张图片
                        "relevant_rules": relevant_rules,
                        "confirmed_violations": confirmed_violations,
                        "suspected_violations": suspected_violations,
                        "rule_encoding": rule_encoding,  # 改为ONE-HOT编码
                        "original_analysis": assistant_content
                    }
                    
                    moe_dataset.append(moe_sample)
                    print(f"成功处理样本，相关规则: {len(relevant_rules)}个，确认违规: {len(confirmed_violations)}个，疑似违规: {len(suspected_violations)}个")
                else:
                    print(f"跳过样本，没有找到assistant回复内容")
        
        # 保存MoE数据集
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(moe_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 处理完成 ===")
        print(f"MoE数据集已保存到: {output_file}")
        print(f"总样本数: {len(moe_dataset)}")
        
        # 统计信息
        self.print_dataset_stats(moe_dataset)
    
    def print_dataset_stats(self, dataset: List[Dict]):
        """打印数据集统计信息"""
        print("\n=== 数据集统计信息 ===")
        
        # 统计每条规则的出现频率
        rule_counts = {rule: 0 for rule in self.all_rules}
        violation_counts = {rule: 0 for rule in self.all_rules}
        
        for sample in dataset:
            for rule in sample['relevant_rules']:
                if rule in rule_counts:
                    rule_counts[rule] += 1
            
            for rule in sample['confirmed_violations'] + sample['suspected_violations']:
                if rule in violation_counts:
                    violation_counts[rule] += 1
        
        print(f"\n最常见的相关规则 (Top 10):")
        sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
        for rule, count in sorted_rules[:10]:
            if count > 0:
                print(f"  {rule}: {count}次")
        
        print(f"\n违反次数最多的规则 (Top 10):")
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        for rule, count in sorted_violations[:10]:
            if count > 0:
                print(f"  {rule}: {count}次")
        
        # 统计ONE-HOT编码的分布
        encoding_stats = [0] * len(self.all_rules)
        for sample in dataset:
            for i, val in enumerate(sample['rule_encoding']):
                encoding_stats[i] += val
        
        print(f"\nONE-HOT编码统计 (出现次数最多的规则):")
        for i, count in enumerate(encoding_stats):
            if count > 0:
                print(f"  规则{i}: {self.all_rules[i]} - 出现{count}次")

# 使用示例
if __name__ == "__main__":
    generator = MoEDatasetGenerator()
    
    # 处理你的JSON文件
    input_file = "/home/baoadmin/LLM/WSC/expand_dataset/202506018dataset2.json"  # 替换为你的输入文件路径
    output_file = "/home/baoadmin/LLM/WSC/expand_dataset/safety_analysis_moe_3.json"  # 输出文件路径
    
    generator.process_json_file(input_file, output_file)
    
    print("\n=== 示例MoE训练样本 ===")
    # 读取并显示一个示例
    with open(output_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
        if samples:
            sample = samples[0]
            print(f"图像: {sample['image']}")
            print(f"相关规则数量: {len(sample['relevant_rules'])}")
            print(f"确认违反: {sample['confirmed_violations']}")
            print(f"疑似违反: {sample['suspected_violations']}")
            print(f"ONE-HOT编码中为1的位置: {[i for i, x in enumerate(sample['rule_encoding']) if x == 1]}")
        else:
            print("没有生成任何样本！")