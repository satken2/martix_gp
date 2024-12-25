# util/npjson.py
import numpy as np
import json

def npobj2json(obj):
    def convert(o):
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert(v) for v in o]
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return o
    return json.dumps(convert(obj))

def json2npobj(json_str):
    def convert(key, o):
        if isinstance(o, dict):
            return {k: (tuple(v) if k == 'shape' else convert(k, v)) for k, v in o.items()}
        elif key == 'value':
            return np.array(o)
        elif isinstance(o, list):
            if all(isinstance(v, (int, float, complex)) for v in o):
                return np.array(o)
            return [convert(None, v) for v in o]
        return o
    return convert(None, json.loads(json_str))
