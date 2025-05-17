module ARDESPOT_Test

using POMDPs, POMDPTools, POMDPModels
using Logging, LoggingExtras
using PyCall
using ..PolicyModule

function neural_solve_and_evaluate(pomdp::TigerPOMDP, model_path; total=50)
    # 加载 Python 模型
    py"""
    from python.feature_policy import load_models
    load_models($model_path)
    """

    # 特征转换函数
    extract_features_jl(belief_vec) = Main.extract_features(belief_vec)
    score_actions_jl(features) = Main.score_actions(features)

    # 初始化策略
    solver = NeuralARDESPOTPlanner(pomdp, extract_features_jl, score_actions_jl)

    total_rewards = []
    for _ in 1:total
        hist = simulate(HistoryRecorder(), pomdp, solver, updater(solver))
        push!(total_rewards, discounted_reward(hist))
    end
    return mean(total_rewards)
end

function run_tests()
    pomdp = TigerPOMDP()
    model_path = joinpath(dirname(dirname(@__DIR__)), "model", "RockSample", "trained_policy")
    avg_reward = neural_solve_and_evaluate(pomdp, model_path)
    @info "Average reward: $avg_reward"
end

end