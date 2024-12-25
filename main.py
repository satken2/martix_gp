# main.py
import numpy as np

# import先はディレクトリ構成に合わせて指定
from neural.nntest1 import NeuralNetTest1
from util.npjson import npobj2json
from util.npjson import json2npobj  # 必要に応じて
# ... 他、必要なところがあれば随時import

if __name__ == '__main__':
    INPUT_SIZE = 10
    OUTPUT_SIZE = 3

    default_obj = {
        'input': {
            'name': 'input',
            'value': np.zeros((INPUT_SIZE,)),
            'shape': (INPUT_SIZE,),
            'init_policy': 'zero',
            'logic': None,
            'fixed': True,
            'used': True,
            'var_score': 0,
        },
        'edge': {
            'name': 'edge',
            'value': np.random.rand(INPUT_SIZE, OUTPUT_SIZE),
            'shape': (INPUT_SIZE, OUTPUT_SIZE),
            'init_policy': 'random',
            'logic': {
                'id': 10, 'type': 2, 'content': 'root', 'shape': (INPUT_SIZE, OUTPUT_SIZE), 'ref': None,
                'args': [
                    {
                        'id': 3, 'type': 2, 'content': 'mul', 'shape': (INPUT_SIZE, OUTPUT_SIZE), 'ref': None,
                        'args': [
                            {
                                'id': 3, 'type': 2, 'content': 'mul', 'shape': (INPUT_SIZE, OUTPUT_SIZE), 'ref': None,
                                'args': [
                                    {'id': 20, 'type': 1, 'content': 'edge', 'shape': (INPUT_SIZE, OUTPUT_SIZE)},
                                    {'id': 5, 'type': 1, 'content': 'update', 'shape': (INPUT_SIZE, OUTPUT_SIZE)},
                                ]
                            },
                            {'id': 5, 'type': 1, 'content': 'sum_ratio', 'shape': (OUTPUT_SIZE,)}
                        ]
                    },
                ]
            },
            'fixed': True,
            'var_score': 0,
            'used': True,
        },
        'output': {
            'name': 'output',
            'value': np.zeros((OUTPUT_SIZE,)),
            'shape': (OUTPUT_SIZE,),
            'init_policy': 'zero',
            'logic': {
                'id': 0, 'type': 2, 'content': 'root', 'shape': (OUTPUT_SIZE,), 'ref': None,
                'args': [
                    {
                        'id': 3, 'type': 2, 'content': 'dot', 'shape': (OUTPUT_SIZE,), 'ref': None,
                        'args': [
                            {'id': 4, 'type': 1, 'content': 'input', 'shape': (INPUT_SIZE,)},
                            {'id': 5, 'type': 1, 'content': 'edge', 'shape': (INPUT_SIZE, OUTPUT_SIZE)},
                        ]
                    },
                ]
            },
            'fixed': True,
            'var_score': 0,
            'used': True,
        },
        'sum_ratio': {
            'name': 'sum_ratio',
            'value': np.random.rand(OUTPUT_SIZE,),
            'shape': (OUTPUT_SIZE,),
            'init_policy': '1',
            'logic': {
                'id': 10, 'type': 2, 'content': 'root', 'shape': (OUTPUT_SIZE,), 'ref': None,
                'args': [
                    {
                        'id': 20, 'type': 2, 'content': 'dev', 'shape': (OUTPUT_SIZE,), 'ref': None,
                        'args': [
                            {'id': 4, 'type': 0, 'content': 1, 'shape': (OUTPUT_SIZE,)},
                            {
                                'id': 20, 'type': 2, 'content': 'sm0', 'shape': (OUTPUT_SIZE,), 'ref': None,
                                'args': [
                                    {'id': 4, 'type': 1, 'content': 'edge', 'shape': (INPUT_SIZE, OUTPUT_SIZE)},
                                ]
                            },
                        ]
                    },
                ]
            },
            'fixed': True,
            'var_score': 0,
            'used': True,
        },
        'update': {
            'name': 'update',
            'value': np.random.rand(INPUT_SIZE, OUTPUT_SIZE),
            'shape': (INPUT_SIZE, OUTPUT_SIZE),
            'init_policy': 'one',
            'logic': {
                'id': 10, 'type': 2, 'content': 'root', 'shape': (INPUT_SIZE, OUTPUT_SIZE), 'ref': None,
                'args': [
                    {'id': 4, 'type': 0, 'content': 1, 'shape': (INPUT_SIZE, OUTPUT_SIZE)},
                ]
            },
            'fixed': True,
            'var_score': 0,
            'used': True,
        },
    }

    # ユーザ入力を受付
    counter = 0
    init_codelist = []
    input_str = ""
    while input_str != "" or counter == 0:
        input_str = input("Program code" + str(counter) + ": ")
        if input_str != "":
            init_codelist.append(input_str)
        counter += 1

    # 入力が無い場合はdefaultを使う
    if len(init_codelist) == 0: 
        init_codelist = [npobj2json(default_obj)]

    # NeuralNetTest1 インスタンスを作成して実行
    EA = NeuralNetTest1(
        codelist=init_codelist,
        default_code=npobj2json(default_obj),
        diversity=5,
        attempts_count=5,
        workers_count=110,
        shuffle_interval=10,
        loops=100,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    )
    EA.exec(loop_count=10000)
