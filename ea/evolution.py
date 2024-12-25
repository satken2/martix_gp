# ea/evolution.py

import copy
import random
import heapq
import traceback
from collections import defaultdict
from datetime import datetime
import string

CONST = 0
VAR = 1
FUNC = 2
GVAL = 3

class BaseEA():
    """
    遺伝的アルゴリズム(進化計算)を行うための基底クラス。
    個体(Worker)リストを管理し、Crossover/Mutationなどで世代交代を進める。
    """
    def __init__(self, codelist=None, default_code="", diversity=5, attempts_count=10,
                 workers_count=10, shuffle_interval=10, loops=10):
        self.workers = []
        self.crossover_ratio = 0.2
        self.tuning_ratio = 0.1
        self.init_codelist = codelist
        self.default_code = default_code
        self.diversity = diversity
        self.attempts_count = attempts_count
        self.workers_count = workers_count
        self.shuffle_interval = shuffle_interval
        self.loops = loops

    def get_worker(self, code=None, majorid=""):
        raise NotImplementedError()

    def get_testdata_list(self):
        raise NotImplementedError()

    def evaluation(self, worker, input_list):
        raise NotImplementedError()

    def get_children(self):
        # 優秀な個体(系統)を抽出し、Crossover/Tuning/Mutationで子を作る
        def append_worker(w_list, new_worker):
            new_worker.common_mutation()
            action_result = new_worker.post_action()
            if action_result and not any(worker.fingerprint == new_worker.fingerprint for worker in w_list):
                w_list.append(new_worker)

        winner_list = self.get_winner_list()

        children = []
        all_variables = {}
        for winner in winner_list:
            append_worker(children, winner)
            for name, variable in winner.variables.items():
                if name not in all_variables:
                    all_variables[name] = [variable]
                else:
                    all_variables[name].append(variable)

        # Crossover
        fixed_var_names = []
        for name, var in winner_list[0].variables.items():
            if var['fixed']:
                fixed_var_names.append(name)
        
        crossover_limit = int((self.workers_count - len(winner_list)) * self.crossover_ratio)
        counter = 0
        while len(children) < (crossover_limit + len(winner_list)) and counter < 100:
            for winner in winner_list:
                try:
                    child = copy.deepcopy(winner)
                    new_variables = {}
                    for name in fixed_var_names:
                        new_variables[name] = copy.deepcopy(random.choice(all_variables[name]))

                    def dfs_add_missing_var(variables, node):
                        if node['type'] == FUNC:
                            for arg in node['args']:
                                dfs_add_missing_var(variables, arg)
                        elif node['type'] == VAR:
                            var_name = node['content']
                            if var_name not in variables:
                                variables[var_name] = copy.deepcopy(random.choice(all_variables[var_name]))
                                dfs_add_missing_var(variables, variables[var_name]['logic'])

                    new_variables_temp = copy.deepcopy(new_variables)
                    for nm, vr in new_variables_temp.items():
                        if vr['logic']:
                            dfs_add_missing_var(new_variables, vr['logic'])

                    child.variables = new_variables
                    append_worker(children, child)
                except Exception as e:
                    print("small Mutation Error!")
                    print(e)
                    traceback.print_exc()
                    exit()
            counter += 1

        # Tuning
        tuning_limit = int((self.workers_count - len(winner_list)) * (self.crossover_ratio + self.tuning_ratio))
        counter = 0
        while len(children) < (tuning_limit + len(winner_list)) and counter < 100:
            for winner in winner_list:
                try:
                    child = copy.deepcopy(winner)
                    child.tuning()
                    append_worker(children, child)
                except Exception as e:
                    print("Tuning Error!")
                    print(e)
                    traceback.print_exc()
                    exit()
            counter += 1

        # Mutation
        while len(children) < self.workers_count:
            for winner in winner_list:
                try:
                    child = copy.deepcopy(winner)
                    child.mutation()
                    append_worker(children, child)
                except Exception as e:
                    print("Major Mutation Error!")
                    print(e)
                    traceback.print_exc()
                    print(winner.variables)
                    exit()

        return children

    def get_winner_list(self):
        major_top_workers = defaultdict(list)
        for worker in self.workers:
            heapq.heappush(major_top_workers[worker.majorid], (-worker.score, worker))
        top_in_each_major = []
        for _, w_heap in major_top_workers.items():
            top_in_each_major.append(w_heap[0][1])
        top_in_each_major.sort(key=lambda w: w.score, reverse=True)
        return top_in_each_major[:self.diversity]

    def exec(self, loop_count=100):
        for _ in range(self.workers_count):
            worker = self.get_worker()
            self.workers.append(worker)
            worker.set_code(self.default_code)
            worker.score = 0

        for index, code in enumerate(self.init_codelist):
            if index < len(self.workers):
                self.workers[index].set_code(code)
                self.workers[index].score = 1

        if len(self.workers[0].variables) == 0:
            print("Empty variable!")
            exit()
            
        exec_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S_")
        print("START: " + exec_id)

        prev_major = ""
        for epoch in range(loop_count):
            try:
                self.exec_epoch(epoch)
                max_worker = max(self.workers, key=lambda worker: worker.score)

                major_change = ""
                if max_worker.majorid != prev_major:
                    major_change = " [TOP LINEAGE CHANGED]"

                file_output = (f"GEN={epoch} "
                               f"PROGRESS={max_worker.get_prog_str()} "
                               f"SCORE={max_worker.score} "
                               f"NODE={max_worker.node_count} "
                               f"MAJOR={max_worker.majorid}"
                               f"{major_change}")
                with open("logs/" + start_timestamp + exec_id + '.txt', 'a') as file:
                    file.write(file_output + "\r\n")
                print(file_output)
                prev_major = max_worker.majorid

                if epoch % self.shuffle_interval == 0:
                    winner_list = self.get_winner_list()
                    content = (f"[TIME={datetime.now().strftime('%Y/%m/%d %H:%M:%S')} "
                               f"EXEC_ID={exec_id} EPOCH={epoch}]\r\n\r\n")
                    for idx_w in range(self.diversity):
                        content += (f"DIV={idx_w} SCORE={winner_list[idx_w].score} "
                                    f"NODE={winner_list[idx_w].node_count} "
                                    f"MAJOR={winner_list[idx_w].majorid} =======================\r\n\r\n")
                        content += winner_list[idx_w].get_code() + "\r\n\r\n"
                    content += "\r\n\r\n"
                    with open('logs/' + exec_id + '.txt', 'a') as file:
                        file.write(content)

            except Exception as e:
                print("Unexpected error!")
                print(e)
                traceback.print_exc()
                print(max_worker.get_code())
                exit()

        print(max_worker.node_count)
        print(max_worker.variables)
        print(max_worker.get_code())

    def exec_epoch(self, epoch):
        self.workers = self.get_children()
        for worker in self.workers:
            worker.reset_score()
            worker.reset_progress()

        for _ in range(self.attempts_count):
            input_list = self.get_testdata_list()
            to_remove = []
            for worker in self.workers:
                worker.init_value()
                try:
                    self.evaluation(worker, input_list)
                except Exception as e:
                    print("Execution error!")
                    traceback.print_exc()
                    to_remove.append(worker)
            for worker in to_remove:
                self.workers.remove(worker)

        for worker in self.workers:
            worker.resize_progress(self.attempts_count * self.loops)
        for worker in self.workers:
            worker.average_score()
