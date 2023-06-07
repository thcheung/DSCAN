import os
import shutil

remove_dirs = []

for root, dirs, files in os.walk("preprocessed"):
    if root.endswith("\processed") or root.endswith("raw"):
        remove_dirs.append(root)

for remove_dir in remove_dirs:
    print(remove_dir)
    shutil.rmtree(remove_dir)
