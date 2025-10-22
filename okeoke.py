import os

def print_tree(startpath, max_depth=None):
    """
    In ra cấu trúc thư mục dưới dạng cây (tree structure)
    Args:
        startpath (str): Đường dẫn thư mục gốc
        max_depth (int, optional): Giới hạn độ sâu (None = in toàn bộ)
    """
    startpath = os.path.abspath(startpath)
    for root, dirs, files in os.walk(startpath):
        # Tính toán độ sâu hiện tại
        level = root.replace(startpath, '').count(os.sep)
        if max_depth is not None and level >= max_depth:
            continue

        indent = '│   ' * level + '├── ' if level > 0 else ''
        print(f"{indent}{os.path.basename(root)}/")

        subindent = '│   ' * (level + 1)
        for f in files:
            print(f"{subindent}├── {f}")


if __name__ == "__main__":
    path = input("Nhập đường dẫn thư mục cần xem (mặc định là thư mục hiện tại): ").strip() or "."
    depth = input("Nhập độ sâu tối đa (Enter nếu muốn in toàn bộ): ").strip()
    depth = int(depth) if depth.isdigit() else None

    print("\n📁 Cấu trúc thư mục:\n")
    print_tree(path, depth)
