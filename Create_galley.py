import os
import random
import shutil

def create_directory_structure(input_folder, output_folder):
    """根据输入文件夹的结构创建输出文件夹的目录结构"""
    for root, dirs, files in os.walk(input_folder):
        # 替换输入文件夹路径为输出文件夹路径
        output_dir = root.replace(input_folder, output_folder, 1)
        # 如果不存在，则创建输出文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

def select_and_copy_images(input_folder, output_folder, num_images=4):
    """从每个二级文件夹中随机选择图片并复制到输出文件夹中相应的目录"""
    for root, dirs, files in os.walk(input_folder):
        # 只处理二级文件夹
        if root.count(os.sep) == input_folder.count(os.sep) + 1:
            # 过滤出所有jpg文件
            jpg_files = [f for f in files if f.lower().endswith('.jpg')]
            if len(jpg_files) > 0:
                # 随机选择指定数量的图片
                selected_files = random.sample(jpg_files, min(num_images, len(jpg_files)))
                # 替换路径，将文件复制到输出文件夹的对应位置
                output_dir = root.replace(input_folder, output_folder, 1)
                for file_name in selected_files:
                    src_file = os.path.join(root, file_name)
                    dst_file = os.path.join(output_dir, file_name)
                    shutil.copy(src_file, dst_file)

def main(input_folder, output_folder):
    # 创建输出文件夹的目录结构
    create_directory_structure(input_folder, output_folder)
    # 随机抽取并复制图片
    select_and_copy_images(input_folder, output_folder)

if __name__ == '__main__':
    input_folder = './output_videos_20240920144211/image'  # 替换为你的输入文件夹路径
    output_folder = './galley'  # 替换为你的输出文件夹路径
    main(input_folder, output_folder)
