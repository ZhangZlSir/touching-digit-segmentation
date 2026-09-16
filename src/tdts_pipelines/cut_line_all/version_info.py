#version_info.py
"""
项目版本信息工具
"""

import sys
import importlib.metadata as metadata

class VersionChecker:
    @staticmethod
    def get_version(package_name):
        """获取包版本"""
        try:
            return metadata.version(package_name)
        except:
            try:
                # 备用方法
                import pkg_resources
                return pkg_resources.get_distribution(package_name).version
            except:
                return "未知"
    
    @staticmethod
    def check_all():
        """检查所有依赖"""
        packages = [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('Pillow', 'PIL'),
            ('tqdm', 'TQDM'),
            ('numba', 'Numba'),
            ('inputimeout', 'InputTimeout'),
        ]
        
        print("\n📦 项目依赖版本:")
        print("-" * 40)
        
        max_len = max(len(name) for _, name in packages)
        
        for pkg, name in packages:
            version = VersionChecker.get_version(pkg)
            print(f"{name:{max_len}}  v{version}")
        
        print(f"\n🐍 Python v{sys.version.split()[0]}")
        print("-" * 40)

# 快速检查
if __name__ == "__main__":
    VersionChecker.check_all()