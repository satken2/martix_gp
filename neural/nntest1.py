# neural/nntest1.py

import random
import copy
import numpy as np

# EA基盤
from ea.evolution import BaseEA
# GPの行列バージョン
from gp.matrix import MatrixGP

SWAP_CHILDREN_RATIO = 0.6
CONST = 0
VAR = 1
FUNC = 2

class NeuralNetTest1(BaseEA):
    """
    BaseEAを継承し、ニューラルネットっぽい構造を評価するクラス。
    input_size, output_sizeなどを受け取ってMatrixGPを生成する。
    """
    def __init__(self, codelist=None, default_code="", diversity=5, attempts_count=10, 
                 workers_count=10, shuffle_interval=10, loops=10,
                 input_size=3, output_size=2):
        super().__init__(codelist=codelist, default_code=default_code,
                         diversity=diversity, attempts_count=attempts_count,
                         workers_count=workers_count, shuffle_interval=shuffle_interval,
                         loops=loops)
        self.input_size = input_size
        self.output_size = output_size

    def get_worker(self):
        # 文字列から8文字抜き出してMajorIDを作る
        import string
        characters = string.ascii_letters + string.digits
        majorid = ''.join(random.choices(characters, k=8))

        return MatrixGP(
            majorid=majorid,
            gval_list=['reward'],
            defined_shapes={'input_size': self.input_size, 'output_size': self.output_size},
            use_gval=False
        )

    def descrete_output2(self, output):
        return_list = copy.deepcopy(output)
        return_list[return_list < 0.2] = 0
        return_list[(return_list >= 0.2) & (return_list < 0.8)] = 0.5
        return_list[return_list >= 0.8] = 1
        return return_list

    def descrete_output(self, output):
        return_list = copy.deepcopy(output)
        return_list[return_list < 0.5] = 0
        return_list[return_list >= 0.5] = 1
        return return_list

    def score_lists(self, list1, list2):
        if len(list1) != len(list2):
            return 0
        matching_count = sum(1 for x, y in zip(list1, list2) if x == y)
        total_elements = len(list1)
        if matching_count == total_elements:
            return 20
        else:
            return int((matching_count / total_elements) * 10)

    def count_output(self, output):
        discrete_list = self.descrete_output(output)
        total = int(np.sum(discrete_list))
        return total

    def get_testdata_list(self):
        seeds = []
        valid1 = {'content': np.random.randint(0, 2, size=(self.input_size, )).astype(np.float64), 'valid': True}
        valid2 = {'content': np.random.randint(0, 2, size=(self.input_size, )).astype(np.float64), 'valid': True}

        for _ in range(self.loops):
            if random.random() < 0.2:
                seeds.append(random.choice([valid1, valid2]))
            else:
                seeds.append({'content': np.random.randint(0, 2, size=(self.input_size, )).astype(np.float64),
                              'valid': False})
        return seeds
    
    def evaluation(self, worker, input_list):
        score = 0
        test_count = int(len(input_list) * 0.2)
        first_score = 0
        last_score = 0
        prev_content = np.zeros((self.input_size,))
        prev_output = np.zeros((self.output_size,))

        for index, data in enumerate(input_list):
            worker.set_values({"input": data['content']})
            worker.exec_calc()
            out = worker.get_values()

            output_array = out['output']
            o_count = self.count_output(output_array)
            o_sum = sum(output_array)

            score_temp = 0

            # 評価1: 0以上1以下に近いほど高得点
            max_distance = None
            for item in output_array:
                if item == 0:
                    continue
                distance_temp = 0
                if item < 0:
                    distance_temp = abs(item)
                if item > 1:
                    distance_temp = abs(item - 1)
                if (max_distance is None) or (distance_temp > max_distance):
                    max_distance = distance_temp
            if max_distance is not None:
                score_temp += max(0, 150 - max_distance * 10)
                worker.progress[1] += max(0, 1 - max_distance)

            # 評価2: validデータなら出力を大きく（o_sumで判定）
            if data['valid']:
                score_temp += min(1, o_sum) * 40
                worker.progress[2] += min(1, max(0, o_sum))
            else:
                worker.progress[2] += 1

            # 評価3: invalidデータなら出力は小さく（o_sumが大きいとペナルティ）
            if not data['valid']:
                score_temp += max(0, o_sum) * -20
                worker.progress[3] += max(0, 1 - max(0, o_sum))
            else:
                worker.progress[3] += 1

            # 評価4: 出力（1の数）が少ないほど良い (o_countが少ないほど加点)
            if o_count != 0:
                score_temp += (self.output_size - o_count) * 10
                if o_count != self.output_size:
                    worker.progress[4] += (self.output_size - o_count) / 2
            else:
                worker.progress[4] += 1

            # 必要なら追加の評価5,6など

            if index < test_count:
                first_score += score_temp
            elif index >= len(input_list) - test_count:
                last_score += score_temp
            score += score_temp

            prev_content = data['content']
            prev_output = output_array

        # ノード数が多いほどペナルティ
        score -= worker.node_count

        worker.score_history.append(score)
