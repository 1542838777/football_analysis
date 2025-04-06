# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
from service.bf import analyze_trading_volume

def main():
    print("足球交易量分析程序")
    print("=" * 50)
    
    try:
        # 检查数据文件是否存在
        if not os.path.exists('bf.csv'):
            print("错误: 未找到数据文件 bf.csv")
            print("请确保数据文件位于程序根目录下")
            return
        
        # 运行分析
        analyze_trading_volume()
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
