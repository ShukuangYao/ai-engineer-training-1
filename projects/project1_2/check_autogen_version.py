#!/usr/bin/env python3
"""
检查 AutoGen 版本的脚本
"""

import sys

def check_autogen_version():
    """检查 AutoGen 版本"""
    try:
        import autogen
        # 方法1: 尝试获取 __version__ 属性
        if hasattr(autogen, '__version__'):
            print(f"AutoGen version (__version__): {autogen.__version__}")
        else:
            print("AutoGen 已安装，但没有 __version__ 属性")

        # 方法2: 尝试获取版本信息
        try:
            import pkg_resources
            version = pkg_resources.get_distribution("pyautogen").version
            print(f"AutoGen version (pkg_resources): {version}")
        except Exception as e:
            print(f"无法通过 pkg_resources 获取版本: {e}")

        # 方法3: 尝试获取模块路径
        print(f"AutoGen 模块路径: {autogen.__file__}")

        # 方法4: 检查是否有 GroupChat 类
        if hasattr(autogen, 'GroupChat'):
            print("✅ 找到 GroupChat 类")
        else:
            print("⚠️ 未找到 GroupChat 类")

        # 方法5: 检查是否有 GroupChatManager 类
        if hasattr(autogen, 'GroupChatManager'):
            print("✅ 找到 GroupChatManager 类")
        else:
            print("⚠️ 未找到 GroupChatManager 类")

    except ImportError as e:
        print(f"❌ AutoGen 未安装: {e}")
        print("\n请运行以下命令安装 AutoGen:")
        print("  pip install pyautogen")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 检查版本时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 50)
    print("AutoGen 版本检查")
    print("=" * 50)
    check_autogen_version()
    print("=" * 50)
