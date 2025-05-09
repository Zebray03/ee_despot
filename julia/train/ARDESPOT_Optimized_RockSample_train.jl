module ARDESPOT_Optimized

using PyCall, JSON3
using POMDPs, POMDPTools, POMDPModels, ARDESPOT
using ParticleFilters
using RockSample
using Random: MersenneTwister
using StaticArrays

export run_optimized_ardespot, run_training

# ========================
# 辅助函数和类型定义
# ========================
function get_pomdp_dims(pomdp::RockSamplePOMDP)
    state_size = length(pomdp.rocks_positions) + 2  # 位置坐标(x,y) + K个岩石状态
    action_size = 5 + length(pomdp.rocks_positions) # 基本移动(4方向) + 采样动作 + K个检测动作
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

function FeatureDrivenPolicy(pomdp::RockSamplePOMDP)
    dims = get_pomdp_dims(pomdp)
    FeatureDrivenPolicy(pomdp, dims.belief_dim, length(pomdp.rocks_positions))
end

function POMDPs.action(policy::FeatureDrivenPolicy, belief)
    particle_list = particles(belief)
    isempty(particle_list) && return rand(actions(policy.pomdp))

    # 构建信念向量
    pos_sum = sum(s.pos for s in particle_list)
    pos_mean = pos_sum ./ length(particle_list)
    rock_probs = [mean(s.rocks[i] for s in particle_list) for i in 1:policy.rock_count]
    belief_vec = vcat(pos_mean..., rock_probs...)

    # 调用Python模型
    features = pycall(Main.extract_features, PyArray, belief_vec)
    action_scores = pycall(Main.score_actions, PyArray, features)

    legal_actions = actions(policy.pomdp)
    return legal_actions[argmax(action_scores)]
end

# ========================
# 数据收集与训练功能
# ========================
function collect_expert_data(pomdp::RockSamplePOMDP; num_episodes=10, max_steps=10)
    expert_data = []
    default_solver = DESPOTSolver(
        bounds = IndependentBounds(-100.0, 20.0, check_terminal=true),
        K = 500,
        max_trials = 10
    )
    expert_planner = solve(default_solver, pomdp)

    # 使用 BootstrapFilter 初始化粒子滤波器
    belief_updater = updater(expert_planner)
    initial_state = initialstate(pomdp)
    filter = BootstrapFilter(pomdp, 1000)

    for _ in 1:num_episodes
        episode_data = []

        # 初始化初始信念
        current_belief = initialize_belief(filter, initial_state)

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

function run_training(;epochs=100, batch_size=32)
    pomdp = RockSamplePOMDP(
        map_size=(5,5),
        rocks_positions=@SVector([(1,1), (3,3), (4,4)]),
        init_pos=RSPos(1,1)
    )

    # 收集专家数据
    @info "Collecting expert data..."
    expert_data = collect_expert_data(pomdp, num_episodes=100)
    training_data = prepare_training_data(expert_data)

    @info "Done"

    @eval Main begin
        # 传输到Python
        global features_train = $training_data.features
        global labels_train = $training_data.labels
    end

    # 执行训练
    @info "Starting training..."
    for epoch in 1:epochs
        loss = Main.train_step(training_data.features, training_data.labels)
        @info "Epoch $epoch: Loss = $loss"
    end

    # 保存模型
    Main.save_models("trained_policy")
end

# ========================
# 主执行函数
# ========================
function run_optimized_ardespot(;max_steps=100)
    pomdp = RockSamplePOMDP(
        map_size=(5,5),
        rocks_positions=@SVector([(1,1), (3,3), (4,4)]),
        init_pos=RSPos(1,1)
    )

    solver = DESPOTSolver(
        bounds = IndependentBounds(-100.0, 20.0, check_terminal=true),
        default_action = FeatureDrivenPolicy(pomdp),
        K = 500,
        rng = MersenneTwister(89757),
        max_trials = 10
    )

    planner = solve(solver, pomdp)
    history = []
    for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=max_steps)
        push!(history, (s, a, o))
    end
    return history
end

end # module