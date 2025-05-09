module ARDESPOT_Optimized
using POMDPs, ARDESPOT, POMDPTools, PyCall, JSON3
using Random: MersenneTwister

export run_optimized_ardespot

function get_pomdp_dims(pomdp::POMDP)
    state_list = states(pomdp) |> collect
    action_list = actions(pomdp) |> collect
    return (
        belief_dim = length(state_list),
        actions_num = length(action_list)
    )
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
    # 将信念转换为概率向量 (适配离散状态空间)
    state_list = states(policy.pomdp) |> collect
    belief_np = [pdf(belief, s) for s in state_list] |> vec

    @assert length(belief_np) == policy.belief_dim "Belief维度不匹配"

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
    pomdp = BabyPOMDP()

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