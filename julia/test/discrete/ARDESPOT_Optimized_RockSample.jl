module ARDESPOT_Optimized
using PyCall, JSON3
using POMDPs, POMDPTools, POMDPModels, ARDESPOT
using ParticleFilters
using RockSample
using Random: MersenneTwister
using StaticArrays

export run_optimized_ardespot

# ========================
# 辅助函数和类型定义
# ========================
function get_pomdp_dims(pomdp::RockSamplePOMDP{K}) where K
    state_size = K + 2  # 位置坐标(x,y) + K个岩石状态
    action_size = 5 + K # 基本移动(4方向) + 采样动作 + K个检测动作
    return (
        belief_dim = state_size,
        actions_num = action_size
    )
end

# ========================
# 自定义策略类型
# ========================
struct FeatureDrivenPolicy{P<:POMDP} <: Policy
    pomdp::P
    belief_dim::Int
    rock_count::Int
end

function FeatureDrivenPolicy(pomdp::RockSamplePOMDP{K}) where K
    dims = get_pomdp_dims(pomdp)
    return FeatureDrivenPolicy(pomdp, dims.belief_dim, K)
end

function POMDPs.action(policy::FeatureDrivenPolicy, belief)
    # 获取粒子集合
    particle_list = particles(belief)
    isempty(particle_list) && return rand(actions(policy.pomdp))

    # 计算位置均值
    pos_sum = sum(s.pos for s in particle_list)
    pos_mean = pos_sum ./ length(particle_list)

    # 计算岩石置信度（仅处理实际存在的岩石）
    rock_probs = [mean(s.rocks[i] for s in particle_list) for i in 1:policy.rock_count]

    # 构建特征向量
    belief_vec = vcat(pos_mean..., rock_probs...)

    # 调用Python接口
    features = pycall(Main.extract_features, PyArray, belief_vec)
    action_scores = pycall(Main.score_actions, PyArray, features)

    # 获取合法动作
    legal_actions = actions(policy.pomdp)
    return legal_actions[argmax(action_scores)]
end

function generate_config(pomdp)
    dims = get_pomdp_dims(pomdp)
    config = Dict(
        "belief_dim" => dims.belief_dim,
        "feature_dim" => 128,  # 增大特征维度适应复杂状态
        "actions_num" => dims.actions_num,
        "map_size" => pomdp.map_size
    )
    JSON3.write("../python/encoder_config.json", config)
end

# ========================
# 主执行函数
# ========================
function run_optimized_ardespot(;
        map_size::Tuple{Int,Int}=(5,5),
        max_steps::Int=20,
        lower_bound=-100.0,
        upper_bound=20.0
    )

    # 初始化POMDP
    pomdp = RockSamplePOMDP(
        map_size=map_size,
        rocks_positions=@SVector([(1,1), (3,3), (4,4)]),  # 明确指定岩石位置
        init_pos=RSPos(1,1)
    )

    generate_config(pomdp)

    # 配置求解器参数
    solver = DESPOTSolver(
        bounds = IndependentBounds(lower_bound, upper_bound),
        default_action = FeatureDrivenPolicy(pomdp),
        K = 500,  # 增加粒子数量
        rng = MersenneTwister(89757),
        tree_in_info = true,
        max_trials = 10_000
    )

    # 生成规划器
    planner = solve(solver, pomdp)

    # 运行模拟
    history = []
    for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=max_steps)
        push!(history, (s, a, o))
    end

    return history
end

end # module