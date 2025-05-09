import os
import sys

sys.path.append(os.path.abspath("../python"))  # 添加Python模块路径

from julia import Main

import julia

# Julia运行时配置
julia.Julia(runtime="julia", compiled_modules=False)

# 加载Python端定义
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将根目录添加到 sys.path
sys.path.insert(0, root_dir)

from python.feature_policy import encoder, scorer

# 加载Julia优化代码
base_dir = os.path.dirname(os.path.abspath(__file__))
julia_code_path = os.path.join(base_dir, "../julia/discrete/ARDESPOT_Optimized_RockSample_train.jl")
#julia_code_path = os.path.join(base_dir, "../julia/continuous/ARDESPOT_Optimized_LightDark1D.jl")
Main.include(julia_code_path)

# 执行优化后的算法
print("Running AR-DESPOT with Feature-Driven Policy...")
history = Main.ARDESPOT_Optimized.run_optimized_ardespot(max_steps=100)

# 打印结果
print("\nSimulation History:")
for i, (s, a, o) in enumerate(history):
    print(f"Step {i+1}:")
    print(f"  State:     {s}")
    print(f"  Action:    {a}")
    print(f"  Observation: {o}\n")