import json
import numpy as np
from dqn_test import test_model


# 加载之前保存的 JSON 文件并按照总奖励排序状态
def load_sorted_states(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    sorted_states = sorted(data, key=lambda x: x['total_reward'])
    return sorted_states


# 选择初始种群
def select_initial_population(sorted_states, population_size):
    print(sorted_states[:population_size])
    return [state['initial_state'] for state in sorted_states[:population_size]]


# 定义适应度函数
def fitness_function(population):
    avg_rewards = []
    for state in population:
        reward = test_model(100, "", state, True)
        avg_rewards.append(reward)
    return avg_rewards


# 交叉操作
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# 变异操作
def mutation(individual, mutation_rate):
    mutated_individual = []
    for gene in individual:
        if np.random.rand() < mutation_rate:
            mutated_gene = np.random.uniform(0, 1)  # 随机生成一个浮点数作为变异
            while not checkMutationReasonable(mutated_gene):
                mutated_gene = np.random.uniform(0, 1)  # 继续尝试生成新的基因直到合理
            mutated_individual.append(mutated_gene)
        else:
            mutated_individual.append(gene)

    # 对变异后的个体进行检查
    while not checkMutationReasonable(mutated_individual):
        for i in range(len(mutated_individual)):
            if np.random.rand() < mutation_rate:
                mutated_gene = np.random.uniform(0, 1)  # 随机生成一个浮点数作为变异
                mutated_individual[i] = mutated_gene
    return mutated_individual


# Check if the seeds generated are under a reasonable situation.
def checkMutationReasonable(state):
    if state[4] > 200 and 320 >= state[3] >= 0 and 300 >= state[6] >= 0:
        return True
    return False


if __name__ == "__main__":
    # 文件路径和种群大小
    file_path = "./test/random_test.json"
    population_size = 30

    # 加载并排序状态
    sorted_states = load_sorted_states(file_path)

    # 选择初始种群
    initial_population = select_initial_population(sorted_states, population_size)

    # 评估初始种群的适应度
    initial_fitness = fitness_function(initial_population)

    # 输出初始种群的适应度
    print("Initial population fitness:", initial_fitness)

    # 遗传算法的一些参数
    num_generations = 10
    crossover_rate = 0.8
    mutation_rate = 0.1

    data = {"generations": []}

    # 迭代进行遗传算法
    for generation in range(num_generations):
        # 选择父代进行交叉和变异
        new_population = []
        while len(new_population) < population_size:
            # 从 initial_population 中随机选择一个个体
            parent1_index = np.random.randint(0, len(initial_population))
            parent1 = initial_population[parent1_index]

            # 从选定的个体中随机选择一个作为父代

            # 选择第二个父代，与第一个父代不能相同，避免自交
            parent2_index = np.random.randint(0, len(initial_population))
            while parent2_index == parent1_index:
                parent2_index = np.random.randint(0, len(initial_population))
            parent2 = initial_population[parent2_index]

            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([mutation(child1, mutation_rate), mutation(child2, mutation_rate)])
            else:
                new_population.append(mutation(parent1, mutation_rate))

        # 更新种群
        initial_population = new_population

        # 计算适应度
        fitness = fitness_function(initial_population)

        # 输出每代的适应度
        print("Generation", generation + 1, "fitness:", fitness)

        data["generations"].append({"population": initial_population, "fitness": fitness})

    # 输出最终种群的适应度
    final_fitness = fitness_function(initial_population)
    print("Final population fitness:", final_fitness)

    # 将字典保存为 JSON 文件
    with open("data/after_mutate.json", 'w') as f:
        json.dump(data, f)
