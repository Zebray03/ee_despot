module ARDESPOT_Optimized
using PyCall, JSON3
using POMDPs, POMDPTools, POMDPModels, ARDESPOT
using Random: MersenneTwister

export run_optimized_ardespot

# 为LightDark1DState定义必要的算术运算
Base.:+(s1::LightDark1DState, s2::LightDark1DState) = LightDark1DState(
    s1.status + s2.status,  # 注意：实际应用中status通常不参与运算
    s1.y + s2.y
)
Base.:-(s::LightDark1DState) = LightDark1DState(-s.status, -s.y)
Base.zero(::Type{LightDark1DState}) = LightDark1DState(0, 0.0)

function get_pomdp_dims(pomdp::POMDP)
    return (belief_dim = 1, actions_num = 2)
end

# ========================
# 1. 定义自定义策略类型
# ========================
struct FeatureDrivenPolicy{P<:POMDP} <: Policy
    pomdp::P
    belief_dim::Int
end

function FeatureDrivenPolicy(pomdp::POMDP)
    dims = get_pomdp_dims(pomdp)
    return FeatureDrivenPolicy(pomdp, dims.belief_dim)
end

function POMDPs.action(policy::FeatureDrivenPolicy, belief)
    particles = [s.y for s in belief]
    y_mean = mean(particles)

    belief_np = [belief_mean]  # 转换为1维数组

    # 调用Python特征提取
    features = pycall(Main.extract_features, PyArray, belief_np)

    # 调用Python动作评分
    action_scores = pycall(Main.score_actions, PyArray, features)

    # 返回最高分动作
    return actions(policy.pomdp)[argmax(action_scores)]
end

function generate_config(pomdp)
    dims = get_pomdp_dims(pomdp)
    config = Dict(
        "belief_dim" => dims.belief_dim,
        "feature_dim" => 64,  # 可自定义或参数化
        "actions_num" => dims.actions_num
    )
    JSON3.write("../python/encoder_config.json", config)
end

# ========================
# 2. 主执行函数
# ========================
function run_optimized_ardespot(;
        max_steps::Int=10,
        lower=-20.0,
        upper=0.0
    )

    # 创建POMDP问题
    pomdp = LightDark1D()

    generate_config(pomdp)

    # 配置求解器参数 (官方推荐方式)
    solver = DESPOTSolver(
        bounds = IndependentBounds(lower, upper),
         default_action = FeatureDrivenPolicy(pomdp),  # 关键：使用自定义策略
         K = 200,              # 确定化观测序列数量
         rng = MersenneTwister(89757),
         tree_in_info = true   # 启用树信息记录
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