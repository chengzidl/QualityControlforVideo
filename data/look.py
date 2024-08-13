import numpy as np

# 加载 .npz 文件
data = np.load('test_data.npz')

# 打印文件中的所有数组名称
print("Keys in the .npz file:", data.keys())

# 打印每个数组的内容
for key in data.keys():
    print(f"Array for {key}:")
    print(data[key])
    