import json
import random
import time

import numpy as np
from scipy.spatial.distance import euclidean, hamming

from DRLFuzz_experiments.flappy_bird.wq.dqn_test import test_model, randomGenerate
from DRLFuzz_experiments.flappy_bird.wq.dqn_test_for_buaa import testInitialState

test_for_one_indivisual = 1

# initial state file path
file_path = "./test/episode_info_random_start.json"

# niching strategy
# msaw: most similar among worst, wams: worst_among_most_similar
n_s_resultarchive = "wams"
n_s_population = "wams"

# distance meature
d_m = "euclidean"

# niching distance threshhold
n_s_t_resultarchive = 20
n_s_t_population = 8

# crowding factor during niching
crowding_factor = 10


class Individual:
    def __init__(self, state, generation, fitness_value=None):
        self.state = state
        self.generation = generation
        self.fitness_value = fitness_value

    def calculate_fitness(self):
        res = test_model(test_for_one_indivisual, self.state)
        self.set_fitness_value(res["avg_award_over_episodes"])
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

    newIn = Individual(mutated_state, generation, 0)
    newIn.fitness_value = newIn.calculate_fitness()
    return newIn


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


def updateCollectionNiching(single_individual, individual_collection):
    random.shuffle(individual_collection)
    # 计算每个组中的个体数量
    group_size = len(individual_collection) // crowding_factor

    # 初始化分组列表
    groups = []

    # 分组
    for i in range(crowding_factor):
        # 获取当前组的起始索引和结束索引
        start_index = i * group_size
        end_index = (i + 1) * group_size

        # 获取当前组的个体
        group = individual_collection[start_index:end_index]

        # 将当前组添加到分组列表中
        groups.append(group)

    # 处理剩余个体
    remaining_individuals = individual_collection[crowding_factor * group_size:]
    for i, ind in enumerate(remaining_individuals):
        groups[i].append(ind)

    # 打印分组结果
    for i, group in enumerate(groups):
        most_similar = group[0]
        most_similar_ditance = 100000
        for index, i_c in enumerate(group):
            # 如果个体与新个体的相似度小于阈值并且个体的适应度低于新个体的适应度
            if similarity(single_individual, i_c, d_m) < most_similar_ditance:
                most_similar = i_c
                # 更新个体集合中的个体为新个体
        if most_similar.fitness_value > single_individual.fitness_value:
            group[index] = single_individual

    flattened_list = [item for sublist in groups for item in sublist]
    return flattened_list


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
    resultArchive = []

    for i in range(iteration):
        # data = test_model(100, randomGenerate())
        data = []
        for j in range(100):
            s = randomGenerate()
            indivisual = Individual(s, 0)
            indivisual.calculate_fitness()
            data.append(indivisual)

        sorted_states = sorted(data, key=lambda x: x.fitness_value)
        current_population = sorted_states

        print("-----------------------iteration {}------------------------".format(i + 1))
        # 迭代进行遗传算法
        for generation in range(generation_iteration):
            print("generation: ", generation + 1, "Start.")
            # 将当前种群按照适应度进行排序
            sorted_population = sorted(current_population, key=lambda x: x.fitness_value)

            # 保留前 10% 的个体到下一代
            next_generation = sorted_population[:population_size // 10]
            population_to_be_replaced = sorted_population[10:]

            # 从当前种群的前 50% 中随机选择两个个体作为父代
            parent1 = random.choice(sorted_population[:population_size // 2])
            parent2 = random.choice(sorted_population[:population_size // 2])

            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
                mutated_child1 = mutation(child1, generation, mutation_rate)
                mutated_child2 = mutation(child2, generation, mutation_rate)

                population_to_be_replaced = updateCollectionNiching(mutated_child1, population_to_be_replaced)
                population_to_be_replaced = updateCollectionNiching(mutated_child2, population_to_be_replaced)

            # 更新种群
            current_population = next_generation + population_to_be_replaced

        # 记录最终种群的状态信息及fitness值
        # final_fitness = getFitnessFromPopulation(current_population)
        print("Generation", generation + 1, "mutated states:", getStatesFromPopulation(current_population))

        # 更新resultArchive
        if len(resultArchive) == 0:
            resultArchive = current_population
        else:
            for c_p in current_population:
                # resultArchive = updateCollectionNiching(c_p, resultArchive)
                resultArchive.extend(current_population)
                # flattened_list = [item for sublist in resultArchive for item in sublist]
                resultArchive = remove_duplicates(resultArchive)
        print("current result archive length: ", len(resultArchive))

    result_archive_dicts = [individual_to_dict(individual) for individual in resultArchive]
    failed_testcases = [individual for individual in result_archive_dicts if individual['fitness_value'] <= -4]
    failed_states = [individual["state"] for individual in failed_testcases]
    failed_testcase_count = len(failed_testcases)
    print("Total count of failed testcases after {} iteration:".format(iteration), failed_testcase_count)

    # 将每一代的状态和适应度写入到 JSON 文件中
    with open("./result/population_data.json", "w") as json_file:
        json.dump(result_archive_dicts, json_file)

    # 将 failed_states 写入到 JSON 文件中
    with open("./result/wq_result.txt", "w") as txt_file:
        json.dump(failed_states, txt_file)


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
    start_time = time.time()
    # iteration, generation, population, crossover_rate, mutation_rate.
    DRLGenetic(100, 10, 100, 0.8, 0.1)
    # check_mutation_reasonable([141.0, 33.0, -102.0, -50.0])
    # print(similarity(Individual([143, 44, -113, 100], 1), Individual([110, 50, -89, 7], 1), d_m))

    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间为：", run_time, "秒")
