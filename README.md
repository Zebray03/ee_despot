## config/

encoder_config.json

```json
{
    "belief_dim":2,
    "hidden_dim":64,
    "action_num":3,
    "feature_dim":128
}
```

## julia/

### RockSample/

policy.jl

```julia
module PolicyModule

using PyCall
using POMDPs, POMDPTools, POMDPModels, RockSample, ARDESPOT
using ParticleFilters
export FeatureDrivenPolicy, NeuralARDESPOTPlanner
using Random: MersenneTwister

# ========== 策略类型定义 ==========
struct FeatureDrivenPolicy{P<:POMDP} <: Policy
    pomdp::P
    belief_dim::Int
    rock_count::Int
end

function FeatureDrivenPolicy(pomdp::RockSamplePOMDP)
    state_size = length(pomdp.rocks_positions) + 2  # 位置(x,y) + 岩石状态
    action_size = 5 + length(pomdp.rocks_positions)
    FeatureDrivenPolicy(pomdp, state_size, length(pomdp.rocks_positions))
end

struct NeuralARDESPOTPlanner{P<:POMDP} <: Policy
    pomdp::P
    extract_features::Function
    score_actions::Function
    belief_updater::Updater
    belief_dim::Int  # 新增维度字段
    action_num::Int  # 新增动作数量字段
end

function POMDPs.updater(planner::NeuralARDESPOTPlanner)
    return planner.belief_updater
end

function NeuralARDESPOTPlanner(pomdp::RockSamplePOMDP, extract_features::Function, score_actions::Function)
    belief_dim, action_num = get_pomdp_dims(pomdp)
    belief_updater = BootstrapFilter(pomdp, 1000) # 使用粒子滤波器（1000个粒子）作为信念更新器
    NeuralARDESPOTPlanner(pomdp, extract_features, score_actions, belief_updater, belief_dim, action_num)
end

function get_pomdp_dims(pomdp::RockSamplePOMDP)
    state_size = length(pomdp.rocks_positions) + 2  # 位置坐标(x,y) + K个岩石状态
    action_size = 5 + length(pomdp.rocks_positions) # 基本移动(4方向) + 采样动作 + K个检测动作
    return (
        belief_dim = state_size,
        actions_num = action_size
    )
end

function POMDPs.action(planner::NeuralARDESPOTPlanner, belief)
    particle_list = support(belief)
    isempty(particle_list) && return rand(actions(planner.pomdp))

    # 确保输入维度正确
    pos_mean = mean(s.pos for s in particle_list)
    rock_probs = [mean(s.rocks[i] for s in particle_list) for i in 1:length(planner.pomdp.rocks_positions)]
    belief_vec = vcat(pos_mean..., rock_probs...)

    # 调用Python模型
    features = planner.extract_features(belief_vec)
    action_scores = planner.score_actions(features)

    # 确保分数维度匹配
    legal_actions = actions(planner.pomdp)
    @assert length(action_scores) == length(legal_actions) "Action score dimension mismatch"

    return legal_actions[argmax(action_scores)]
end

end # module
```

RockSample_test.jl

```julia
module ARDESPOT_Test

using POMDPs, POMDPTools
using RockSample
using ParticleFilters
using StaticArrays
using Logging, LoggingExtras
using PyCall
np = pyimport("numpy")
using Base.Threads

export run_tests

include("policy.jl")

logger = TeeLogger(
    global_logger(),
    FileLogger("neural_ardespot.log"; append=true)
)
global_logger(logger)

# 更新测试函数
function neural_solve_and_evaluate(pomdp::RockSamplePOMDP, model_prefix; max_steps=100, total=100)
    # 加载Python模型
    py"""
    from python.feature_policy import load_models
    load_models($model_prefix)
    """

    function extract_features_jl(belief_vec::Vector{Float64})
        belief_np = np.array(belief_vec, dtype=np.float32)
        features = Main.extract_features(belief_np)
        return convert(Vector{Float32}, features)
    end

    function score_actions_jl(features::Vector{Float32})
        features_np = np.array(features, dtype=np.float32)
        action_scores = Main.score_actions(features_np)
        @info "$action_scores"
        return convert(Vector{Float64}, action_scores)
    end

    # 初始化策略
    solver = PolicyModule.NeuralARDESPOTPlanner(pomdp, extract_features_jl, score_actions_jl)

    # 并行测试逻辑
    total_rewards = Vector{Float64}(undef, total)
    lock = ReentrantLock()

    @threads for i in 1:total
        try
            hist = simulate(HistoryRecorder(max_steps=max_steps), pomdp, solver, updater(solver))
            total_rewards[i] = discounted_reward(hist)
        catch e
            @warn "Trial $i failed: $e"
            total_rewards[i] = 0.0
        end
        @info "$total_rewards"
    end


    return mean(total_rewards)
end

# 更新测试配置
function run_tests()
    pomdp_base = RockSamplePOMDP(
        map_size = (5,5),
        rocks_positions = @SVector([(2,2), (3,3), (4,4)]),
        init_pos = RSPos(1,1)
    )

    model_path = joinpath(dirname(dirname(@__DIR__)), "model", "RockSample")

    # 基础测试
    @info "Base configuration test..."
    base_reward = neural_solve_and_evaluate(pomdp_base, model_path, total=100)
    @info "Base average reward: $base_reward"

    # 粒子数量对比测试
    particle_configs = [
        (100, 20.0, -10),
        (500, 20.0, -10),
        (1000, 20.0, -10)
    ]

    results = Dict{Int, Float64}()
    for (pc, se, brp) in particle_configs
        @info "Testing $pc particles..."
        pomdp_test = RockSamplePOMDP(
            map_size = (5,5),
            rocks_positions = @SVector([(2,2), (3,3), (4,4)]),
            init_pos = RSPos(1,1),
            sensor_efficiency = se,
            bad_rock_penalty = brp
        )
        avg = neural_solve_and_evaluate(pomdp_test, model_path, total=50)
        results[pc] = avg
    end

    @info "Final results:" results
end

end # module
```

RockSample_train.jl

```julia
module ARDESPOT_Train

using PyCall, JSON3
using POMDPs, POMDPTools, POMDPModels, ARDESPOT
using ParticleFilters
using RockSample
using Random
using StaticArrays
using ..PolicyModule: FeatureDrivenPolicy

export run_training

# ========================
# 数据收集与训练功能
# ========================
function collect_expert_data(; num_episodes, max_steps)
    map_size = (5,5)
    expert_data = []
    for _ in 1:num_episodes
        pomdp = RockSamplePOMDP(
            map_size::Tuple{Int, Int} = map_size,
            rocks_positions::SVector{K,RSPos} = @SVector([(2,2), (3,3), (4,4)]),
            init_pos::RSPos = RSPos(rand(1:map_size[1]), rand(1:map_size[2])),
            sensor_efficiency::Float64 = 20.0,
            bad_rock_penalty::Float64 = -10,
            good_rock_reward::Float64 = 10,
            step_penalty::Float64 = 0,
            sensor_use_penalty::Float64 = 0,
        )

        default_solver = DESPOTSolver(
            bounds = IndependentBounds(-100.0, 20.0, check_terminal=true),
            K = 500,
            max_trials = 25
        )
        expert_planner = solve(default_solver, pomdp)

        # 使用 BootstrapFilter 初始化粒子滤波器
        belief_updater = updater(expert_planner)
        initial_state = initialstate(pomdp)
        filter = BootstrapFilter(pomdp, 1000)

        episode_data = []

        # 初始化初始信念
        current_belief = initialize_belief(filter, initial_state)
        @info "$current_belief"

        for (s, a, o, r) in stepthrough(pomdp, expert_planner, "s,a,o,r", max_steps=max_steps)
            # 使用滤波器更新信念
            current_belief = update(filter, current_belief, a, o)

            particle_list = particles(current_belief)
            pos_mean = mean(s.pos for s in particle_list)
            rock_probs = [mean(s.rocks[i] for s in particle_list) for i in 1:length(pomdp.rocks_positions)]
            belief_vec = vcat(pos_mean..., rock_probs...)
            action_idx = findfirst(isequal(a), actions(pomdp))
            push!(episode_data, (belief_vec, action_idx))
        end
        push!(expert_data, episode_data)
    end
    return expert_data
end

function prepare_training_data(expert_data)
    features = []
    labels = []
    for episode in expert_data
        for (bvec, aidx) in episode
            push!(features, bvec)
            push!(labels, aidx - 1)
        end
    end
    (features = hcat(features...)', labels = labels)
end

function run_training()
    # 收集专家数据
    @info "Collecting expert data..."
    expert_data = collect_expert_data(num_episodes=100, max_steps=25)
    training_data = prepare_training_data(expert_data)

    @eval Main begin
        global features_train = $training_data.features
        global labels_train = $training_data.labels
    end
end

end # module
```

## model/

### RockSample/

RockSample_encoder.pth

RockSample_decoder.pth

## python/

__init__.py

```python

```

config.py

```python
import os
import json

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config_path(filename="encoder_config.json"):
    """获取配置文件路径"""
    return os.path.join(PROJECT_ROOT,
                        "config",
                        filename)


def load_config():
    """加载模型配置"""
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        return json.load(f)


def get_julia_code_path(subdir, script_name):
    """获取Julia代码路径"""
    return os.path.join(PROJECT_ROOT,
                        "julia",
                        subdir,
                        script_name)


def get_model_path(subdir):
    """获取模型保存路径"""
    return os.path.join(PROJECT_ROOT, "model", subdir)

```

data.py

```python
import torch
from torch.utils.data import Dataset, DataLoader


class RockSampleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_dataloader(features, labels, batch_size=32):
    """创建数据加载器"""
    dataset = RockSampleDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
```

feature_policy.py

```python
# feature_policy.py
from .config import get_model_path
from .model import load_models

# 仅加载预训练模型
encoder, scorer = load_models(get_model_path("RockSample"))
```

julia_integration.py

```python
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

```

model.py

```python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from python.config import load_config


class ResidualBlock(nn.Module):
    """残差块，用于增强网络深度"""

    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        out = F.relu(out)
        return out


class BeliefEncoder(nn.Module):
    """增强版信念编码器，包含残差结构、注意力机制和LSTM时序处理"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_res_blocks=3, seq_len=10):
        super(BeliefEncoder, self).__init__()

        # 输入层
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        # 残差块堆栈
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # 输入处理
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = F.relu(x)

        # 通过残差块
        for block in self.res_blocks:
            x = block(x)

        # 应用注意力机制
        attn_weights = self.attention(x)
        x = x * attn_weights

        # 输出层
        x = self.fc_out(x)
        return x


class ActionScorer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_num):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_num),
        )

    def forward(self, x):
        return self.fc(x)


def create_models(input_dim, hidden_dim, feature_dim, action_num):
    """创建编码器和动作评分器模型"""
    encoder = BeliefEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=feature_dim)
    scorer = ActionScorer(feature_dim=feature_dim, hidden_dim=hidden_dim, action_num=action_num)
    return encoder, scorer


def load_models(model_dir):
    """加载预训练模型"""
    config = load_config()
    encoder, scorer = create_models(
        config["belief_dim"],
        config["hidden_dim"],
        config["feature_dim"],
        config["action_num"]
    )

    encoder_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_encoder.pth")
    scorer_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_scorer.pth")

    # 加载模型参数后，显式转换为 float32
    encoder.load_state_dict(
        {k: v.float() for k, v in torch.load(encoder_path, map_location='cpu').items()},  # 强制权重为 float32
        strict=True
    )
    scorer.load_state_dict(
        {k: v.float() for k, v in torch.load(scorer_path, map_location='cpu').items()},  # 强制权重为 float32
        strict=True
    )
    encoder.eval()
    scorer.eval()
    return encoder, scorer


def save_models(encoder, scorer, model_dir):
    """保存模型到指定目录"""
    os.makedirs(model_dir, exist_ok=True)  # 确保目录存在

    # 构建完整文件路径
    encoder_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_encoder.pth")
    scorer_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_scorer.pth")

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(scorer.state_dict(), scorer_path)


def extract_features(encoder, belief_np):
    """统一特征提取接口"""
    belief_tensor = torch.from_numpy(belief_np).float()
    with torch.no_grad():
        features = encoder(belief_tensor).numpy()
    return features


def score_actions(scorer, features_np):
    """统一动作评分接口"""
    features_tensor = torch.from_numpy(features_np).float()
    with torch.no_grad():
        scores = scorer(features_tensor).numpy()
    return scores

```

solver.py

```python
import torch
from python.model import create_models, load_models
from python.config import load_config, get_model_path
from julia import Main


class NeuralARDESPOTSolver:
    """神经增强的ARDESPOT求解器类"""

    def __init__(self, pomdp, model_path=None):
        self.pomdp = pomdp
        config = load_config()

        # 初始化模型
        if model_path:
            self.encoder, self.scorer = load_models(model_path)
        else:
            self.encoder, self.scorer = create_models(
                config["belief_dim"],
                config["hidden_dim"],
                config["feature_dim"],
                config["action_num"]
            )

        # 暴露接口给Julia
        Main.extract_features = lambda x: self.encoder(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        Main.score_actions = lambda x: self.scorer(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    def solve(self, pomdp):
        """返回使用神经增强策略的规划器"""
        from julia import Main
        return Main.eval("""
        NeuralPlanner = NeuralARDESPOTPlanner($pomdp, $(self.encoder), $(self.scorer))
        """)
```

trainer.py

```python
# trainer.py
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .model import create_models, save_models
from .data import prepare_dataloader
from .config import load_config, get_model_path


def train_step(features, labels, encoder, scorer, optimizer, criterion, batch_size):
    """执行一个训练步骤"""
    dataloader = prepare_dataloader(features, labels, batch_size=batch_size)
    encoder.train()
    scorer.train()
    total_loss = 0.0

    for batch_features, batch_labels in dataloader:
        optimizer.zero_grad()
        features = encoder(batch_features)
        scores = scorer(features)
        loss = criterion(scores, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(epochs, batch_size, problem, patience=10):
    """训练模型主函数"""
    config = load_config()

    # 创建模型
    encoder, scorer = create_models(
        config["belief_dim"],
        config["hidden_dim"],
        config["feature_dim"],
        config["action_num"]
    )

    # 一次性获取所有数据
    from julia import Main
    features = np.array(Main.features_train)
    labels = np.array(Main.labels_train)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(scorer.parameters()), lr=1e-4)

    # 初始化损失历史
    loss_history = []
    best_loss = float('inf')
    best_epoch = 0

    # 训练循环
    for epoch in range(epochs):
        loss = train_step(features, labels, encoder, scorer, optimizer, criterion, batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        # 更新损失历史
        loss_history.append(loss)
        if len(loss_history) > patience:
            loss_history.pop(0)

        # 检查是否有改善
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch

        # 检查是否早停
        if len(loss_history) >= patience:
            min_loss = min(loss_history)
            max_loss = max(loss_history)
            if (max_loss - min_loss) / min_loss <= 0.05:
                print(f"Early stopping at epoch {epoch + 1}. Loss variation within 10% in {patience} epochs.")
                break

    # 保存模型
    model_path = get_model_path(subdir=problem)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # 确保目录存在
    save_models(encoder, scorer, model_path)

    # 返回训练好的模型实例
    return encoder, scorer
```

## scripts/

test.py

```python
import logging

import torch
from julia import Main

from python.config import get_model_path
from python.julia_integration import init_julia
from python.model import load_models

problem = 'RockSample'

def run_neural_ardespot_test():
    # 初始化Julia环境
    julia_main = init_julia(mode='test', problem=problem)

    # 加载训练好的模型
    model_path = get_model_path(subdir=problem)
    encoder, scorer = load_models(model_path)

    # 绑定模型函数到Julia
    device = next(encoder.parameters()).device
    julia_main.extract_features = lambda x: encoder(
        torch.tensor(x, dtype=torch.float32)
        .unsqueeze(0).to(device)).squeeze(0).cpu().detach().numpy()

    julia_main.score_actions = lambda x: scorer(
        torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy()

    # 执行Julia测试
    julia_main.eval("""
            ARDESPOT_Test.run_tests()
            """)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    run_neural_ardespot_test()

```

train.py

```julia
import sys

import torch

sys.path.append("..")
from python.julia_integration import init_julia
from python.trainer import train_model

problem = 'RockSample'


def train():
    # 初始化Julia环境
    Main = init_julia(mode='train', problem=problem)

    # 收集数据
    Main.eval("ARDESPOT_Train.run_training()")

    # 执行训练并保存模型
    encoder, scorer = train_model(epochs=10000, batch_size=32, problem=problem)

    # 更新Julia接口使用训练后的模型
    from julia import Main
    Main.extract_features = lambda x: encoder(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    Main.score_actions = lambda x: scorer(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    print("Training completed. Models saved.")


if __name__ == "__main__":
    train()

```

