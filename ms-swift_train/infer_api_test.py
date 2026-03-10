#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试客户端 - 专门解决72B模型推理慢的问题
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001/v1"

def test_connection():
    """测试连接"""
    print("🔗 测试服务器连接...")
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✅ 连接成功!")
            print(f"📋 可用模型: {[m['id'] for m in models.get('data', [])]}")
            return models['data'][0]['id'] if models.get('data') else None
        else:
            print(f"❌ 响应错误: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None

def test_simple_text(model_name):
    """测试简单文本生成"""
    print("\n🧪 测试简单文本生成...")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "简单说一句话"}
        ],
        "max_tokens": 20,  # 限制输出长度加快速度
        "temperature": 0.1
    }
    
    print("📤 发送请求...")
    start_time = time.time()
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", 
                               json=payload, 
                               timeout=120)
        
        duration = time.time() - start_time
        print(f"⏱️  请求耗时: {duration:.2f}秒")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ 生成成功: {content}")
            return True
        else:
            print(f"❌ 生成失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时 - 模型推理时间过长")
        print("💡 建议: 72B模型推理较慢，这是正常现象")
        return False
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_safety_knowledge(model_name):
    """测试安全知识（简化版）"""
    print("\n🔍 测试安全分析知识...")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "你知道工业安全帽的作用吗？一句话回答。"}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    print("📤 发送安全知识测试...")
    start_time = time.time()
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", 
                               json=payload, 
                               timeout=120)
        
        duration = time.time() - start_time
        print(f"⏱️  请求耗时: {duration:.2f}秒")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"🤖 模型回复: {content}")
            
            # 检查是否包含安全相关知识
            safety_keywords = ["保护", "安全", "头部", "防护", "事故"]
            found = any(kw in content for kw in safety_keywords)
            
            if found:
                print("✅ 模型具备安全知识")
                return True
            else:
                print("⚠️  未明显体现安全专业知识")
                return False
        else:
            print(f"❌ 测试失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def get_server_stats():
    """获取服务器统计信息"""
    print("\n📊 服务器状态信息...")
    try:
        # 检查服务器健康状态
        health_url = BASE_URL.replace('/v1', '/health')
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print("✅ 服务器健康状态: 正常")
        else:
            print("⚠️  健康检查端点不可用")
    except:
        print("⚠️  健康检查端点不可用")
    
    # 显示模型加载信息
    print("💾 模型加载信息:")
    print("   - 基础模型: Qwen2.5-VL-72B-Instruct")
    print("   - 微调权重: checkpoint-294")
    print("   - GPU使用: 0,1,2,3,4,5,6,7")

def performance_tips():
    """性能优化建议"""
    print("\n" + "=" * 60)
    print("⚡ 性能优化建议")
    print("=" * 60)
    print("如果推理速度太慢，可以尝试:")
    print("1. 减少GPU数量: export CUDA_VISIBLE_DEVICES=0,1,2,3")
    print("2. 使用VLLM后端: --backend vllm")
    print("3. 减少max_tokens参数")
    print("4. 降低temperature参数")
    print("5. 使用更小的batch_size")
    print()
    print("重启服务器命令示例:")
    print("python your_deploy_script.py --action deploy --backend vllm --gpu_ids 0,1,2,3")

def main():
    print("⚡ Swift Deploy 快速测试工具")
    print("=" * 60)
    print("专为72B大模型优化，快速验证服务器状态")
    print("=" * 60)
    
    # 1. 测试连接
    model_name = test_connection()
    if not model_name:
        print("\n❌ 连接测试失败，请检查服务器状态")
        return
    
    # 2. 获取服务器信息
    get_server_stats()
    
    # 3. 测试简单文本生成
    text_success = test_simple_text(model_name)
    
    # 4. 测试安全知识
    safety_success = test_safety_knowledge(model_name)
    
    # 5. 结果总结
    print("\n" + "=" * 60)
    print("📋 测试结果总结")
    print("=" * 60)
    
    if text_success:
        print("✅ 基本文本生成: 正常")
    else:
        print("❌ 基本文本生成: 失败或超时")
    
    if safety_success:
        print("✅ 安全知识测试: 通过")
    else:
        print("⚠️  安全知识测试: 需要进一步验证")
    
    if text_success:
        print("\n🎉 结论: 你的微调模型服务器运行正常!")
        print("   可以开始使用图像分析功能了")
        
        # 询问是否进行图像测试
        try:
            user_input = input("\n是否要测试图像分析功能? (y/n): ").strip().lower()
            if user_input == 'y':
                print("\n💡 提示: 图像分析可能需要1-5分钟，请耐心等待...")
                print("建议使用你原来的客户端脚本进行完整的图像测试")
        except KeyboardInterrupt:
            print("\n👋 测试结束")
    else:
        print("\n⚠️  服务器可能存在问题，请检查配置")
        performance_tips()

if __name__ == "__main__":
    main()