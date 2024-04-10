import json
import random
import numpy as np

from DRLFuzz_experiments.flappy_bird.wq.dqn_test import test_model

test_for_one_indivisual = 1


class Individual:
    def __init__(self, state, generation, fitness_value=None):
        self.state = state
        self.generation = generation
        self.fitness_value = fitness_value

    def calculate_fitness(self):
        res = test_model(test_for_one_indivisual, self.state)
        return res["avg_award_over_episodes"]

    def set_fitness_value(self, value):
        self.fitness_value = value


# 交叉操作
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, len(parent1.state))
    child1_state = parent1.state[:crossover_point] + parent2.state[crossover_point:]
    child2_state = parent2.state[:crossover_point] + parent1.state[crossover_point:]
    return Individual(child1_state, max(parent1.generation, parent2.generation)), Individual(child2_state,
                                                                                             max(parent1.generation,
                                                                                                 parent2.generation))


# 变异操作
def mutation(individual, mutation_rate):
    mutated_state = list(individual.state)  # 转换成列表类型
    for i in range(len(mutated_state)):
        if np.random.rand() < mutation_rate:
            while True:
                mutation_amount = np.random.uniform(-0.1, 0.1)  # 在[-0.1, 0.1]范围内生成一个随机数
                mutated_gene = mutated_state[i] + mutation_amount
                mutated_state[i] = mutated_gene  # 将变异后的基因应用到个体的状态中
                if check_mutation_reasonable(mutated_state):
                    break

    return Individual(mutated_state, individual.generation)


# Check if the seeds generated are under a reasonable situation.
def check_mutation_reasonable(state):
    state = list(state) if isinstance(state, tuple) else state
    if 25 <= state[0] <= 192 and 25 <= state[1] <= 192 and -120 <= state[2] <= -75 and -56 <= state[3] <= 10:
        return True
    return False


# 加载之前保存的 JSON 文件并按照总奖励排序状态
def load_sorted_states(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        # data = data[:100]
    sorted_states = sorted(data, key=lambda x: x['avg_award_over_episodes'])
    return sorted_states


# 选择初始种群
def select_initial_population(sorted_states, population_size):
    return [Individual(state['initial_state'], 1, state['avg_award_over_episodes']) for state in
            sorted_states[:population_size]]


# 定义适应度函数
def fitness_function(population):
    avg_rewards = []
    for individual in population:
        print("fitness_function: calculate avg fitness over episodes,using initial state:", individual.state)
        res = test_model(test_for_one_indivisual, individual.state)
        individual.fitness_value = res["avg_award_over_episodes"]
        avg_rewards.append(res["avg_award_over_episodes"])
    return avg_rewards


file_path = "./test/episode_info_random_start.json"


#DRLGenetic
def DRLGenetic(iteration, generation_iteration, population_size, crossover_rate, mutation_rate):
    # 加载并排序状态
    sorted_states = load_sorted_states(file_path)
    test_initial = []
    for i in sorted_states:
        test_initial.append(i["avg_award_over_episodes"])
    print("Initial Population Fitness Value: (Sort)", test_initial)

    # 选择初始种群
    initial_population = select_initial_population(sorted_states, population_size)

    # 评估初始种群的适应度(avg)
    initial_fitness = []
    for initial_individual in initial_population:
        initial_fitness.append(initial_individual.fitness_value)

    # 输出初始种群的适应度
    print("Initial population fitness:", initial_fitness)

    current_population = initial_population

    # 用于存储每一代的状态和适应度
    generation_data = []

    # 迭代进行遗传算法
    for generation in range(generation_iteration):
        print("generation: ", generation + 1, "Start.")
        # 将当前种群按照适应度进行排序
        sorted_population = sorted(current_population, key=lambda x: x.fitness_value)

        # 保留前 10% 的个体到下一代
        next_generation = sorted_population[:population_size // 10]

        # 选择父代进行交叉和变异
        new_population = []
        while len(new_population) < population_size - len(next_generation):
            # 从当前种群的前 50% 中随机选择两个个体作为父代
            parent1 = random.choice(sorted_population[:population_size // 2])
            parent2 = random.choice(sorted_population[:population_size // 2])

            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([mutation(child1, mutation_rate), mutation(child2, mutation_rate)])
            else:
                new_population.append(mutation(parent1, mutation_rate))

        # 更新种群
        current_population = next_generation + new_population

        # 计算适应度
        newFitness = fitness_function(current_population)

        # 输出每代的适应度
        print("Generation", generation + 1, "fitness:", newFitness)
        print("Generation", generation + 1, "mutated states:", getStatesFromPopulation(current_population))
        print("Generation", generation + 1, "Ends.")

        # 记录每一代的状态和适应度
        generation_data.append({
            "generation": generation + 1,
            "fitness": newFitness,
            "states": getStatesFromPopulation(current_population)
        })

    # 输出最终种群的适应度
    final_fitness = getFitnessFromPopulation(current_population)
    print("Final population fitness:", final_fitness)

    # 将每一代的状态和适应度写入到 JSON 文件中
    with open("./data/generation_data.json", "w") as json_file:
        json.dump(generation_data, json_file)


def getStatesFromPopulation(population):
    stateList = []
    for individual in population:
        stateList.append(individual.state)
    return stateList


def getFitnessFromPopulation(population):
    fitnessList = []
    for individual in population:
        fitnessList.append(individual.fitness_value)
    return fitnessList


if __name__ == "__main__":
    # iteration, generation, population, crossover_rate, mutation_rate.
    DRLGenetic(1000, 50, 10, 0.8, 0.1)
    # check_mutation_reasonable([141.0, 33.0, -102.0, -50.0])
