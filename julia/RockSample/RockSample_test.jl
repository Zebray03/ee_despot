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

# 测试函数
function evaluate(pomdp::RockSamplePOMDP, model_prefix; max_steps=100, total=100)
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
        return convert(Vector{Float64}, action_scores)
    end

    solver = PolicyModule.NeuralARDESPOTPlanner(pomdp, extract_features_jl, score_actions_jl)

    total_rewards = Vector{Float64}(undef, total)
    lock = ReentrantLock()

    @threads for i in 1:total
        try
            initial_belief = initialize_belief(solver.updater, initialstate(pomdp))
            hist = simulate(HistoryRecorder(max_steps=max_steps), pomdp, solver, solver.updater)
            total_rewards[i] = discounted_reward(hist)
        catch e
            @warn "Trial $i failed: $e"
            total_rewards[i] = 0.0
        end
        @info "Total_rewards: $total_rewards"
    end
    return mean(total_rewards)
end

function run_tests()
    pomdp_base = RockSamplePOMDP(
        map_size = (5,5),
        rocks_positions = @SVector([(2,2), (3,3), (4,4)]),
        init_pos = RSPos(1,1)
    )

    model_path = joinpath(dirname(dirname(@__DIR__)), "model", "RockSample")

    @info "Base configuration test..."
    base_reward = evaluate(pomdp_base, model_path, total=100)
    @info "Base average reward: $base_reward"

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
        avg = evaluate(pomdp_test, model_path, total=50)
        results[pc] = avg
    end

    @info "Final results:" results
end

end # module