import random

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt


class SLGA:

    def __init__(self, FJSP_table):
        self.max_gen = 50
        self.population_size = 100
        self.f_c = 0.8
        self.f_m = 0.2
        self.FJSP_table = FJSP_table
        self.machines_count = len(self.FJSP_table[0][0])
        self.seq_list = self.get_seq_list()
        self.population = self.init_population()
        times = []
        for job in self.FJSP_table:
            for op in job:
                for t in op:
                    if t is not None:
                        times.append(t)
        self.max_slot = sum(times)
        self.selection_problems_probability_list = None
        self.best_chromosome = self.population[0]
        self.best_fitness = self.fitness(self.best_chromosome)

    def get_seq_list(self):
        result = []
        for job_i, job in enumerate(self.FJSP_table):
            for _ in job:
                result.append(job_i)

        return result

    def get_machines_time(self, solution):
        assigned_list = self.get_assigned_list(solution)
        machine_time = [[] for _ in range(self.machines_count)]
        machine_ops_list = []
        for machine in range(self.machines_count):
            machine_ops_list.append([item for item in assigned_list if item["machine"] == machine])
        assigned_list = []
        max_length_machines = len(max(machine_ops_list, key=lambda item: len(item)))
        for index in range(max_length_machines):
            for machine in machine_ops_list:
                try:
                    assigned_list.append(machine[index])
                except Exception as e:
                    pass
        for item in assigned_list:
            end_time = item["time"]
            slot = 0
            while slot != end_time:
                add_job = True
                for machine in range(self.machines_count):
                    try:
                        if machine != item["machine"] and machine_time[machine][len(machine_time[item["machine"]])] == \
                                item[
                                    "job"]:
                            machine_time[item["machine"]].append(None)
                            end_time += 1
                            add_job = False
                            break

                    except:
                        pass
                if add_job:
                    machine_time[item["machine"]].append(item["job"])
                slot += 1

        return machine_time

    def time_of_solution(self, solution):
        machine_time = self.get_machines_time(solution)
        return len(max(machine_time, key=lambda item: len(item)))

    def get_assigned_list(self, solution):
        result = []
        for index in range(len(solution[0])):
            machine = solution[1][index]
            job = solution[0][index]
            operation = solution[0][0:index].count(solution[0][index])
            result.append(({
                "machine": machine,
                "job": job,
                "operation": operation,
                "time": self.FJSP_table[job][operation][machine]
            }))

        return result

    def get_machines_from_seq_list(self, seq_list):
        result = []
        for index in range(len(seq_list)):
            machines = []
            job = seq_list[index]
            operation = seq_list[0:index].count(seq_list[index])
            for machine in range(self.machines_count):
                if self.FJSP_table[job][operation][machine] is not None:
                    machines.append(machine)

            if random.random() < self.f_m:
                result.append(random.choice(machines))
            else:
                result.append(min(machines, key=lambda item: self.FJSP_table[job][operation][item]))
        return result

    def init_population(self):
        population = []
        for i in range(self.population_size):
            new_seq_list = self.seq_list.copy()
            random.shuffle(new_seq_list)
            machines = self.get_machines_from_seq_list(new_seq_list)
            population.append([new_seq_list, machines])

        return population

    def fitness(self, solution):
        return -self.time_of_solution(solution)

    def get_time(self, job, operation, machine):
        return self.FJSP_table[job][operation][machine]

    def exec_time(self):

        times = []
        for machine_index, machine_op in enumerate():
            times.append(sum([self.FJSP_table[t[0] - 1][t[1] - 1][machine_index] for t in machine_op]))

        return max(times)

    def update_selection_problems_probability_list(self):
        sum_fitness = sum([self.fitness(c) for c in self.population])
        if sum_fitness != 0:
            self.selection_problems_probability_list = [self.fitness(c) / sum_fitness for c in self.population]
        else:
            self.selection_problems_probability_list = [1 / len(self.population) for c in range(self.population_size)]

    def fps(self):
        # return an index
        return npr.choice(self.population_size, p=self.selection_problems_probability_list)

    def generate_offsprings(self, parents_pool):
        offsprings = []
        for parent in parents_pool:
            job_seq = parent[0].copy()
            if random.random() <= self.f_c:
                [first, last] = [random.randint(0, len(job_seq)), random.randint(0, len(job_seq))]
                sub_list = job_seq[first:last]
                random.shuffle(sub_list)
                job_seq[first:last] = sub_list
                random.shuffle(job_seq)

            offsprings.append([job_seq, self.get_machines_from_seq_list(job_seq)])
        return offsprings

    def run(self):

        for generation in range(self.max_gen):
            self.update_selection_problems_probability_list()
            parents_pool = [self.population[self.fps()] for i in range(self.population_size)]
            random.shuffle(parents_pool)
            offsprings = self.generate_offsprings(parents_pool=parents_pool)
            self.population = offsprings
            best_current_chromosome = max(self.population, key=lambda item: self.fitness(item))
            best_current_fitness = self.fitness(best_current_chromosome)
            if best_current_fitness > self.best_fitness:
                self.best_chromosome = best_current_chromosome
                self.best_fitness = best_current_fitness


def show_plot(machine_time, job_size):
    fig, gnt = plt.subplots()
    gnt.set_ylim(1, len(machine_time) + 1)
    gnt.set_xlim(0, len(max(machine_time, key=lambda item: len(item))))
    gnt.set_xlabel('times')
    gnt.set_ylabel('machines')
    # gnt.set_yticks([15, 25, 35])
    # gnt.set_yticklabels([str(i) for i in range(len(machine_time))])

    # Setting graph attribute
    gnt.grid(True)
    for machine in range(len(machine_time)):
        for i in range(len(machine_time[machine])):
            if machine_time[machine][i] is None:
                continue
            color = ['orange', 'blue', 'red', 'green', 'gray', 'purple'][machine_time[machine][i]]
            gnt.broken_barh([(i, 1)], (machine + 1, 0.5), facecolors=(f"tab:{color}"))

    r = np.linspace(0, 1, 10)
    for i in range(job_size):
        plt.plot(0, 0, color=['orange', 'blue', 'red', 'green', 'gray', 'purple'][i], label=str(i + 1))
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    problem_table1 = \
        [
            [
                [2, None, 5],
                [None, 1, 4]
            ],
            [
                [2, 3, None],
                [4, 2, 1]
            ],
            [
                [2, 3, None],
                [2, None, 5]
            ],
            [
                [4, None, 2],
                [None, 3, 5],
                [1, None, 2]
            ]

        ]
    problem_table2 = \
        [
            [
                [2, None, 7, 5],
                [2, 1, 4, None],
                [1, 10, None, None]
            ],
            [
                [2, 5, None, 4],
                [4, 2, 1, 2],
                [5, 5, 5, 6]
            ],
            [
                [2, 3, None, 1],
                [2, 6, 5, 10],
            ],
            [
                [4, None, 2, 3],
                [None, 3, 5, 3],
                [1, None, 2, 4],
                [2, None, 3, 4]
            ],
            [
                [7, 2, None, 2],
                [5, 15, 5, None]
            ]

        ]

    problem_table3 = \
        [
            [
                [20, None, 70, 15],
                [17, 10, 22, None],
                [19, 15, None, None]
            ],
            [
                [2, 5, None, 4],
                [4, 2, 1, 2],
                [5, 5, 5, 6]
            ],
            [
                [21, 31, None, 12],
                [22, 16, 15, 20],
            ],
            [
                [4, None, 2, 3],
                [None, 3, 5, 3],
                [1, None, 2, 4],
                [2, None, 3, 4]
            ]

        ]
    problems = [problem_table1, problem_table2, problem_table3]
    for problem in problems:
        slga = SLGA(FJSP_table=problem)
        slga.run()
        print(slga.best_fitness)
        for i in slga.get_machines_time(slga.best_chromosome):
            print(i)

        show_plot(slga.get_machines_time(slga.best_chromosome), len(problem))
