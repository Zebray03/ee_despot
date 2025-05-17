module ARDESPOT_Train

using POMDPs, POMDPTools, POMDPModels, ARDESPOT
using PyCall, JSON3, ParticleFilters
using Random: MersenneTwister
using ..PolicyModule: FeatureDrivenPolicy

export run_training

# 定义POMDP维度结构体
struct POMDPDimensions
    belief_dim::Int
    actions_num::Int
end

# 获取POMDP维度信息
function get_pomdp_dims(pomdp::POMDP)
    S = statetype(pomdp)
    A = actiontype(pomdp)
    belief_dim = length(initialstate(pomdp))
    actions_num = length(actions(pomdp))
    return POMDPDimensions(belief_dim, actions_num)
end

# 转换信念状态为向量表示
function belief_to_vector(b::ParticleCollection, pomdp::POMDP)
    state_space = states(pomdp)
    n_states = length(state_space)
    vec = zeros(n_states)

    # 计算每个状态的粒子比例
    for s in state_space
        idx = findfirst(isequal(s), state_space)
        if idx !== nothing
            vec[idx] = weight(b, s)
        end
    end

    return vec
end

function generate_config(pomdp::POMDP)
    dims = get_pomdp_dims(pomdp)
    config = Dict(
        "belief_dim" => dims.belief_dim,
        "hidden_dim" => 64,
        "feature_dim" => 128,
        "action_num" => dims.actions_num
    )
    JSON3.write("../config/encoder_config.json", config)
end

function collect_expert_data(pomdp::POMDP, solver::DESPOTSolver; num_episodes=100)
    expert_data = []
    expert_policy = solve(solver, pomdp)
    belief_updater = updater(expert_policy)

    for episode in 1:num_episodes
        episode_data = []

        for (s, b, a, o, r, sp, bp) in stepthrough(
                pomdp,
                expert_policy,
                belief_updater,
                "s,b,a,o,r,sp,bp",
                max_steps=20,
                rng=MersenneTwister(episode))

            # 处理不同类型的信念表示
            if b isa ParticleCollection
                belief_vec = belief_to_vector(b, pomdp)
            else
                belief_vec = convert(AbstractVector, b)
            end

            action_idx = findfirst(isequal(a), actions(pomdp))
            push!(episode_data, (belief_vec, action_idx))
        end

        push!(expert_data, episode_data)
    end

    return expert_data
end

function prepare_training_data(expert_data)
    features = Vector{Float64}[]
    labels = Int[]

    for episode in expert_data
        for (bvec, aidx) in episode
            push!(features, bvec)
            push!(labels, aidx - 1)
        end
    end

    return (features = hcat(features...)', labels = labels)
end

function run_training(
        max_steps::Int=10,
        lower=-20.0,
        upper=0.0)

    pomdp = TigerPOMDP()
    generate_config(pomdp)

    solver = DESPOTSolver(
        bounds = IndependentBounds(lower, upper),
        K = 200,
        rng = MersenneTwister(89757),
        tree_in_info = true
    )

    expert_data = collect_expert_data(pomdp, solver)
    training_data = prepare_training_data(expert_data)

    @eval Main begin
        global features_train = $training_data.features
        global labels_train = $training_data.labels
    end

    return training_data
end

end