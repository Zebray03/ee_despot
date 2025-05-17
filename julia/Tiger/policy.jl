module PolicyModule

using POMDPs, POMDPTools, POMDPModels
using PyCall
export FeatureDrivenPolicy, NeuralARDESPOTPlanner

# ========== 策略类型定义 ==========
struct FeatureDrivenPolicy{P<:POMDP} <: Policy
    pomdp::P
    belief_dim::Int
end

function FeatureDrivenPolicy(pomdp::TigerPOMDP)
    belief_dim = 2  # 左/右门有老虎的概率
    FeatureDrivenPolicy(pomdp, belief_dim)
end

struct NeuralARDESPOTPlanner{P<:POMDP} <: Policy
    pomdp::P
    extract_features::Function
    score_actions::Function
    belief_updater::Updater
    belief_dim::Int
    action_num::Int
end

function POMDPs.updater(planner::NeuralARDESPOTPlanner)
    return planner.belief_updater
end

function NeuralARDESPOTPlanner(pomdp::TigerPOMDP, extract_features::Function, score_actions::Function)
    belief_dim = 2
    action_num = 3  # 动作：左开、右开、听
    belief_updater = DiscreteUpdater(pomdp)
    NeuralARDESPOTPlanner(pomdp, extract_features, score_actions, belief_updater, belief_dim, action_num)
end

function POMDPs.action(planner::NeuralARDESPOTPlanner, belief)
    belief_vec = [pdf(belief, :left), pdf(belief, :right)]
    features = planner.extract_features(belief_vec)
    action_scores = planner.score_actions(features)
    legal_actions = actions(planner.pomdp)
    return legal_actions[argmax(action_scores)]
end

end