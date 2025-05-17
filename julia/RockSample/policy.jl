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
    updater::Updater
    belief_dim::Int
    action_num::Int
end


function NeuralARDESPOTPlanner(pomdp::RockSamplePOMDP, extract_features::Function, score_actions::Function)
    belief_dim, action_num = get_pomdp_dims(pomdp)
    updater = BootstrapFilter(pomdp, 1000)
    return NeuralARDESPOTPlanner(pomdp, extract_features, score_actions, updater, belief_dim, action_num)
end

function get_pomdp_dims(pomdp::RockSamplePOMDP)
    state_size = length(pomdp.rocks_positions) + 2 # 位置坐标(x,y) + K个岩石状态
    action_size = length(pomdp.rocks_positions) + 5 # 基本移动(4方向) + 采样动作 + K个检测动作
    return (
        belief_dim = state_size,
        actions_num = action_size
    )
end

function POMDPs.action(planner::NeuralARDESPOTPlanner, belief)
    particle_list = support(belief)
    isempty(particle_list) && return rand(actions(planner.pomdp))

    pos_mean = mean(s.pos for s in particle_list)
    rock_probs = [mean(s.rocks[i] for s in particle_list) for i in 1:length(planner.pomdp.rocks_positions)]
    belief_vec = vcat(pos_mean..., rock_probs...)

    # 继续处理特征提取和动作评分
    features = planner.extract_features(belief_vec)
    action_scores = planner.score_actions(features)
    legal_actions = actions(planner.pomdp)

    return legal_actions[argmax(action_scores)]
end


end # module