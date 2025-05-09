from julia import Julia, Main
import os

from python.config import get_julia_code_path, PROJECT_ROOT

_julia_initialized = False


def init_julia():
    """初始化Julia环境"""
    global _julia_initialized
    if not _julia_initialized:
        # 设置Julia项目路径
        project_dir = os.path.join(PROJECT_ROOT, "julia")
        Julia(runtime="julia", compiled_modules=False, project=project_dir)

        # 加载Julia模块
        from julia import Pkg
        Pkg.activate(project_dir)

        Pkg.add("POMDPs")
        Pkg.add("POMDPTools")
        Pkg.add("POMDPModels")
        Pkg.add("RockSample")
        Pkg.add("ParticleFilters")
        Pkg.add("ARDESPOT")
        Pkg.add("StaticArrays")
        Pkg.precompile()

        # 加载自定义模块
        julia_code_path = get_julia_code_path("ARDESPOT_Optimized_RockSample_train.jl")
        julia_code_path = julia_code_path.replace("\\", "/")
        Main.eval(f'include("{julia_code_path}")')
        Main.eval("using .ARDESPOT_Optimized")

        _julia_initialized = True
    return Main