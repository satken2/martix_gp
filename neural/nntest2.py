import random
import copy
from evolution import BaseEA
import matrixgp
import numpy as np
import string

SWAP_CHILDREN_RATIO = 0.6
CONST = 0
VAR = 1
FUNC = 2

class NeuralNetTest2(BaseEA):
    def __init__(self, codelist=None, default_code="", diversity=5, attempts_count=10, workers_count=10, shuffle_interval=10, loops=10):
        super().__init__(codelist=codelist, default_code=default_code, diversity=diversity, attempts_count=attempts_count, workers_count=workers_count, shuffle_interval=shuffle_interval)
        self.loops = loops

    def get_worker(self):
        # 系統IDの決定
        characters = string.ascii_letters + string.digits
        majorid = ''.join(random.choices(characters, k=8))
        return matrixgp.MatrixGP(majorid=majorid)

    # numpy配列で与えられるoutputの要素を以下の規則で数える
    # 0.2未満は0と見なす
    # 0.2～0.8 は0.5とみなす
    # 0.8以上は1とみなす
    def descrete_output2(self, output):
        # output配列の要素に対して条件を適用して値を変更
        return_list = copy.deepcopy(output)
        return_list[return_list < 0.2] = 0
        return_list[(return_list >= 0.2) & (return_list < 0.8)] = 0.5
        return_list[return_list >= 0.8] = 1
        return return_list

    def descrete_output(self, output):
        # output配列の要素に対して条件を適用して値を変更
        return_list = copy.deepcopy(output)
        return_list[return_list < 0.5] = 0
        return_list[return_list >= 0.5] = 1
        return return_list

    def score_lists(self, list1, list2):
        if len(list1) != len(list2):
            return 0  # リストの長さが異なる場合は比較不可能として0点を返す

        matching_count = sum(1 for x, y in zip(list1, list2) if x == y)
        total_elements = len(list1)
        
        if matching_count == total_elements:
            return 20  # 全ての要素が一致する場合は特別ボーナス
        else:
            # それ以外の場合は、一致する要素の割合に基づいてスコアを計算する
            return int((matching_count / total_elements) * 10)

    def count_output(self, output):
        descrete_list = self.descrete_output(output)
        # 変換された値の合計を計算し、整数に変換して返す
        total = int(np.sum(descrete_list))
        return total

    def get_testdata_list(self):
        seeds = [
            {'content': np.array([0, 0, 0]), 'valid': False},
            {'content': np.array([0, 0, 0]), 'valid': False},
            {'content': np.array([0, 0, 0]), 'valid': False},
            {'content': np.array([0, 0, 0]), 'valid': False},
            {'content': np.array([0, 0, 0]), 'valid': False},
            {'content': np.array([1, 0, 0]), 'valid': True},
            {'content': np.array([0, 1, 0]), 'valid': True},
            {'content': np.array([1, 0, 0]), 'valid': True},
            {'content': np.array([0, 1, 0]), 'valid': True},
            {'content': np.array([1, 0, 0]), 'valid': True},
            {'content': np.array([0, 1, 0]), 'valid': True},
            {'content': np.array([0, 0, 1]), 'valid': False},
        ]
        return_list = []
        for _  in range(self.loops):
            return_list.append(random.choice(seeds))
        
        return return_list
    
    def evaluation(self, worker, input_list):
        score = 0
        prev_sum = 0
        test_count = int(len(input_list) * 0.2)
        first_score = 0
        last_score = 0
        for index, data in enumerate(input_list):
            worker.set_values({"input": data['content']})
            worker.exec_calc()
            out = worker.get_values()
            
            score_temp = 0
            # bandwithが変化すればOKという緩い評価
            sum_temp = np.sum(out['bandwidth'])
            if sum_temp != prev_sum:
                score_temp += 1
            prev_sum = sum_temp

            # 出力が0か1ならプラス
            if data['valid']:
                if self.count_output(out['output']) == 1:
                    score_temp += 3
                elif self.count_output(out['output']) == 2:
                    score_temp -= 3
            if not data['valid']:
                if self.count_output(out['output']) != 0:
                    score_temp -= 3
            
            if index < test_count:
                first_score += score_temp
            elif index >= len(input_list) - test_count:
                last_score += score_temp
            score += score_temp

        # 最初より最後の方が改善してれば加算
        score += (last_score - first_score) * 10

        worker.score_history.append(score)

