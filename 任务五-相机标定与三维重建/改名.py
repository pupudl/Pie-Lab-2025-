import os
from pathlib import Path


def rename_photos(folder_path, output_folder=None, prefix="img_", start_num=1):
    # 如果指定了输出文件夹，则创建它
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中所有的图片文件
    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(extensions)])

    # 检查文件数量是否匹配
    if len(files) != 121:
        print(f"警告：文件夹中有{len(files)}张照片，但预期是73张")

    # 重命名文件
    for i, filename in enumerate(files, start=start_num):
        # 获取文件扩展名
        ext = Path(filename).suffix.lower()

        # 生成新文件名（如img_01.jpg）
        new_name = f"{prefix}{i:02d}{ext}"

        # 原始文件完整路径
        old_path = os.path.join(folder_path, filename)

        # 新文件完整路径
        if output_folder:
            new_path = os.path.join(output_folder, new_name)
        else:
            new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_name}")


if __name__ == "__main__":
    # 使用示例
    photo_folder = "D:/colmap_project/test2/input/111"  # 替换为你的照片文件夹路径
    output_folder = "D:/colmap_project/test2/input/room_photos"  # 设为None则原地重命名

    rename_photos(
        folder_path=photo_folder,
        output_folder=output_folder,
        prefix="img_",
        start_num=1
    )

    print("重命名完成！")