# gp/matrix.py
import numpy as np
import random
from functools import reduce

from gp.base import GPBase, CONST, VAR, FUNC, GVAL
from util import filter as ft

class MatrixGP(GPBase):
    """
    行列演算を扱う拡張クラス。
    add/mul/dev/dotなどの演算関数をFUNC_MASTERに登録し、シェイプ判定も行う。
    """
    def __init__(self, code=None, majorid="", gval_list=[], defined_shapes={}, use_gval=False):
        super().__init__(majorid=majorid, gval_list=gval_list, defined_shapes=defined_shapes, use_gval=use_gval)
        self.FUNC_MASTER = {
            'root': {'name': 'root', 'func': self.root, 'reset': False, 'arg_count': 1, 'shapeRef': self.shape_root},
            'add': {'name': 'add', 'func': self.add, 'reset': False, 'arg_count': 2, 'shapeRef': self.shape_add},
            'mul': {'name': 'multiple', 'func': self.multiple, 'reset': False, 'arg_count': 2, 'shapeRef': self.shape_add},
            'dev': {'name': 'devide', 'func': self.devide, 'reset': False, 'arg_count': 2, 'shapeRef': self.shape_add},
            'dot': {'name': 'dot', 'func': self.dot, 'reset': False, 'arg_count': 2, 'shapeRef': self.shape_dot},
            'nrm': {'name': 'normalize', 'func': self.normalize, 'reset': False, 'arg_count': 1, 'shapeRef': self.shape_root},
            'clm': {'name': 'clip_min', 'func': self.clip_min, 'reset': True, 'arg_count': 2, 'shapeRef': self.shape_clip},
            'clx': {'name': 'clip_max', 'func': self.clip_max, 'reset': True, 'arg_count': 2, 'shapeRef': self.shape_clip},
            'bin': {'name': 'binarize', 'func': self.binarize, 'reset': False, 'arg_count': 1, 'shapeRef': self.shape_root},
            'sm0': {'name': 'h_sum', 'func': self.sum_0, 'reset': False, 'arg_count': 1, 'shapeRef': self.shape_sum0},
            'sm1': {'name': 'v_sum', 'func': self.sum_1, 'reset': False, 'arg_count': 1, 'shapeRef': self.shape_sum1},
        }

    # 実際の演算関数
    def normalize(self, data, shape=None):
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.divide(data - mean, std, where=std!=0)
        except FloatingPointError:
            return data

    def add(self, a, b, shape=None):
        try:
            value = a + b
            return ft.threshold(ft.cap(value, 1000000, 1000000), -10000000, -10000000)
        except FloatingPointError:
            return a

    def multiple(self, a, b, shape=None):
        try:
            value = a * b
            return ft.threshold(ft.cap(value, 1000000, 1000000), -10000000, -10000000)
        except FloatingPointError:
            return a

    def devide(self, a, b, shape=None):
        try:
            return a / ft.remove_zero(b)
        except FloatingPointError:
            return a

    def dot(self, a, b, shape=None):
        return np.dot(a, b)

    def clip_min(self, a, threshold_value, shape=None):
        return np.maximum(a, threshold_value)

    def clip_max(self, a, threshold_value, shape=None):
        return np.minimum(a, threshold_value)

    def binarize(self, a, shape=None):
        return np.where(a == 0, 0, 1)

    def sum_0(self, input_array, shape=None):
        return np.sum(input_array, axis=0)

    def sum_1(self, input_array, shape=None):
        result = np.sum(input_array, axis=1)
        if np.shape(result) == (1,):
            return result[0]
        return result

    # 以下シェイプ判定用
    def shape_root(self, output_shape, input_lineups, pinned_shape=[None]):
        return self.filter_pin([[output_shape]], pinned_shape)

    def shape_clip(self, output_shape, input_lineups, pinned_shape=[None, None]):
        valid_combinations = [[output_shape, ()]]
        return self.filter_pin(valid_combinations, pinned_shape)

    def shape_add(self, output_shape, input_lineups, pinned_shape=[None, None]):
        if len(np.shape(output_shape)) == 0:
            return [(), ()]
        variables = []
        if len(np.shape(output_shape)) == 2:
            variables.append((np.shape(output_shape)[0], 1))
            variables.append((np.shape(output_shape)[1],))
        variables.append(())
        variables.append(output_shape)

        valid_combinations = [
            [output_shape, random.choice(variables)],
            [random.choice(variables), output_shape],
        ]
        return self.filter_pin(valid_combinations, pinned_shape)

    def shape_dot(self, output_shape, input_lineups, pinned_shape=[None, None]):
        valid_combinations = []
        for i in range(len(input_lineups)):
            for j in range(len(input_lineups)):
                if i != j:
                    a = np.zeros(input_lineups[i])
                    b = np.zeros(input_lineups[j])
                    try:
                        result = np.dot(a, b)
                        if result.shape == output_shape:
                            valid_combinations.append((input_lineups[i], input_lineups[j]))
                    except ValueError:
                        continue
        return self.filter_pin(valid_combinations, pinned_shape)

    def shape_sum0(self, output_shape, input_lineups, pinned_shape=[None]):
        valid_combinations = []
        if len(output_shape) > 1:
            return None

        if len(output_shape) == 0:
            for item in input_lineups:
                if len(item) == 1:
                    valid_combinations.append([item])
        elif len(output_shape) == 1:
            for item in input_lineups:
                if len(item) == 2 and output_shape[0] == item[1]:
                    valid_combinations.append([item])
        return self.filter_pin(valid_combinations, pinned_shape)

    def shape_sum1(self, output_shape, input_lineups, pinned_shape=[None]):
        valid_combinations = []
        if len(output_shape) == 1:
            for item in input_lineups:
                if len(item) == 2 and item[0] == output_shape[0]:
                    valid_combinations.append([item])
        return self.filter_pin(valid_combinations, pinned_shape)

    def filter_pin(self, valid_combinations, pinned_shape):
        filtered_combinations = []
        for combo in valid_combinations:
            matched = True
            for idx, ps in enumerate(pinned_shape):
                if ps is not None and np.shape(combo[idx]) != ps:
                    matched = False
                    break
            if matched:
                filtered_combinations.append(combo)
        if filtered_combinations:
            return random.choice(filtered_combinations)
        else:
            return None
