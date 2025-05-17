from julia import Julia, Main
import os

from python.config import get_julia_code_path, PROJECT_ROOT

_julia_initialized = False


def init_julia(mode, problem):
    """初始化Julia环境"""
    global _julia_initialized
    if not _julia_initialized:
        # 设置Julia项目路径
        project_dir = os.path.join(PROJECT_ROOT, "julia", problem)
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

        # 加载策略模块
        common_path = get_julia_code_path(subdir=problem, script_name="policy.jl")
        common_path = common_path.replace("\\", "/")
        Main.eval(f'include("{common_path}")')
        Main.eval("using .PolicyModule")

        if mode == 'train':
            code_path = get_julia_code_path(subdir=problem, script_name=f'{problem}_Train.jl')
            code_path = code_path.replace("\\", "/")
            Main.eval(f'include("{code_path}")')
            Main.eval("using .ARDESPOT_Train")
        else:
            code_path = get_julia_code_path(subdir=problem, script_name=f'{problem}_Test.jl')
            code_path = code_path.replace("\\", "/")
            Main.eval(f'include("{code_path}")')
            Main.eval("using .ARDESPOT_Test")

        _julia_initialized = True
    return Main
