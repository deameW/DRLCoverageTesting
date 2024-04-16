import json
import random
import numpy as np
from scipy.spatial.distance import euclidean, hamming

from DRLFuzz_experiments.flappy_bird.wq.dqn_test import test_model

test_for_one_indivisual = 1

# initial state file path
file_path = "./test/episode_info_random_start.json"

# niching strategy
# msaw: most similar among worst, wams: worst_among_most_similar
n_s = "wams"

# distance meature
d_m = "euclidean"

# niching distance threshhold
n_s_t = 50


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
def mutation(individual, generation, mutation_rate):
    mutated_state = list(individual.state)  # 转换成列表类型
    for i in range(len(mutated_state)):
        if np.random.rand() < mutation_rate:
            while True:
                mutation_amount = np.random.uniform(-0.1, 0.1)  # 在[-0.1, 0.1]范围内生成一个随机数
                mutated_gene = mutated_state[i] + mutation_amount
                mutated_state[i] = mutated_gene  # 将变异后的基因应用到个体的状态中
                if check_mutation_reasonable(mutated_state):
                    break

    return Individual(mutated_state, generation, individual.generation)


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
        # print("fitness_function: calculate avg fitness over episodes,using initial state:", individual.state)
        res = test_model(test_for_one_indivisual, individual.state)
        individual.fitness_value = res["avg_award_over_episodes"]
        avg_rewards.append(res["avg_award_over_episodes"])
    return avg_rewards


# d_m: distance measure
def similarity(indivisual1, indivisual2, d_m):
    state1 = indivisual1.state
    state2 = indivisual2.state

    if d_m is None:
        d_m = 'euclidean'
    # import pdb; pdb.set_trace()
    if d_m == 'euclidean':
        return euclidean(state1, state2)
    if d_m == 'hamming':
        return hamming(state1, state2)

    raise ValueError("Unknown distance measure {}".
                     format(d_m))


# 小生境遗传算法淘汰策略
"""
    对于DRLGenetic的每一次调用，最后种群中会有100个个体，第一回合的将加入到一个resultArchive中，
    接下来的每一回合，生成的最终种群使用niching_strategy
"""


def updateCollectionNiching(single_individual, individual_collection, threshold):
    bak = []
    # 遍历个体集合中的每个个体
    if n_s == "wams":
        flag = False
        for index, i_c in enumerate(individual_collection):
            # 如果个体与新个体的相似度低于阈值并且个体的适应度低于新个体的适应度
            if similarity(single_individual, i_c,
                          d_m) <= threshold and single_individual.fitness_value < i_c.fitness_value:
                # 更新个体集合中的个体为新个体
                flag = True
                individual_collection[index] = single_individual

        # 如果找不到相似并且fitness <= -4，直接加入
        if flag == False and single_individual.fitness_value <= -4:
            bak.append(single_individual)

    individual_collection.extend(bak)

    if n_s == "msaw":  # most similar among worst
        sortedCollection = sorted(individual_collection, key=lambda x: x['fitness_value'],
                                  reverse=True)  # fitness大的需要被替换
        for index, s_c in enumerate(sortedCollection):
            if index < 10:  # 前10个fitness最大的替换
                if similarity(single_individual, s_c, d_m) < threshold:
                    sortedCollection[index] = single_individual

    # 排重
    return remove_duplicates(individual_collection)


def remove_duplicates(individual_collection):
    unique_collection = []
    for indivisual in individual_collection:
        state_exists = False
        for unique_individual in unique_collection:
            if indivisual.state == unique_individual.state:
                state_exists = True
                break
        if not state_exists:
            unique_collection.append(indivisual)
    return unique_collection


def individual_to_dict(individual):
    return {
        "state": individual.state,
        "generation": individual.generation,
        "fitness_value": individual.fitness_value
    }


# DRLGenetic
def DRLGenetic(iteration, generation_iteration, population_size, crossover_rate, mutation_rate):
    # 加载并排序状态
    sorted_states = load_sorted_states(file_path)
    test_initial = []
    for i in sorted_states:
        test_initial.append(i["avg_award_over_episodes"])
    # print("Initial Population Fitness Value: (Sort)", test_initial)

    # 选择初始种群
    initial_population = select_initial_population(sorted_states, population_size)

    # 评估初始种群的适应度(avg)
    initial_fitness = []
    for initial_individual in initial_population:
        initial_fitness.append(initial_individual.fitness_value)

    # 输出初始种群的适应度
    # print("Initial population fitness:", initial_fitness)

    current_population = initial_population

    # 用于存储每一代的状态和适应度
    generation_data = []
    # resultArchive = initial_population
    resultArchive = []

    for i in range(iteration):
        print("-----------------------iteration {}------------------------".format(i + 1))
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
                    new_population.extend(
                        [mutation(child1, generation, mutation_rate), mutation(child2, generation, mutation_rate)])
                else:
                    new_population.append(mutation(parent1, generation, mutation_rate))

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

        # 更新resultArchive
        if len(resultArchive) == 0:
            resultArchive = current_population
        else:
            for c_p in current_population:
                resultArchive = updateCollectionNiching(c_p, resultArchive, n_s_t)
        print("current result archive length: ", len(resultArchive))
        # print("Final population fitness:", final_fitness)

    result_archive_dicts = [individual_to_dict(individual) for individual in resultArchive]
    failed_testcases = [individual for individual in result_archive_dicts if individual['fitness_value'] <= -4]
    failed_testcase_count = len(failed_testcases)
    print("Total count of failed testcases after {} iteration:".format(iteration), failed_testcase_count)

    # 将每一代的状态和适应度写入到 JSON 文件中
    with open("./data/generation_data.json", "w") as json_file:
        json.dump(result_archive_dicts, json_file)


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
    DRLGenetic(1000, 5, 20, 0.8, 0.1)
    # check_mutation_reasonable([141.0, 33.0, -102.0, -50.0])
    # print(similarity(Individual([143, 44, -113, 100], 1), Individual([110, 50, -89, 7], 1), d_m))
