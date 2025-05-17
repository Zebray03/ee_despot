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