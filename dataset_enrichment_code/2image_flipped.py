import os
from PIL import Image

def flip_images_horizontal(source_dir, output_dir):
    """
    将源目录下的所有图片进行水平翻转并保存到输出目录
    
    Args:
        source_dir (str): 源图片目录路径
        output_dir (str): 输出目录路径
    """
    
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取所有图片文件
    image_files = []
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(supported_formats):
            image_files.append(filename)
    
    if not image_files:
        print(f"在目录 {source_dir} 中没有找到支持的图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始水平翻转处理...")
    
    processed_count = 0
    error_count = 0
    
    for filename in image_files:
        try:
            # 读取图片
            source_path = os.path.join(source_dir, filename)
            image = Image.open(source_path)
            
            # 水平翻转
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 保存到新目录，文件名不变
            output_path = os.path.join(output_dir, filename)
            flipped_image.save(output_path)
            
            processed_count += 1
            print(f"处理完成: {filename}")
            
        except Exception as e:
            error_count += 1
            print(f"处理失败: {filename} - 错误: {str(e)}")
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 张图片")
    print(f"处理失败: {error_count} 张图片")
    print(f"输出目录: {output_dir}")

def main():
    # 设置路径
    source_dir = '/data/LLM/WSC/dataset_video/extracted_frames'
    output_dir = '/data/LLM/WSC/dataset_video/flipped_frames'
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 执行水平翻转
    flip_images_horizontal(source_dir, output_dir)

if __name__ == "__main__":
    main()