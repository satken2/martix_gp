# gp/base.py
import numpy as np
import random
import copy
import string
import hashlib
import time

from util.npjson import npobj2json, json2npobj

# 定数識別用の定義
CONST = 0  # 定数ノード
VAR = 1    # 変数ノード
FUNC = 2   # 関数ノード (子を持つ)
GVAL = 3   # グローバル変数ノード (別枠の定義リストを参照)

# 変数の基本テンプレート例。GPBaseで新たに変数を作成するときなどに参照
VARIABLE_TEMPLATE = {
    'value': 0,
    # ロジック = 「どの演算をして value を得るか」をツリー状に保持
    'logic': {
        'type': FUNC,       # FUNC or VAR or CONST
        'content': 'root',  # 実際の演算内容 (例: 'root', 'add'など)
        'ref': None,        # 実際の関数参照をPythonレベルで保存
        'args': [
            {'type': CONST, 'content': 0}
        ]
    },
    'fixed': False,         # Trueなら変異・交叉で変更しない
    'used': True,           # 使われているかどうか
    'unused_count': 0,      # 使われていない期間をカウント (一定以上で削除)
}

class GPBase():
    """
    行列演算をベースとしたGP(遺伝的プログラミング)の基底クラス。

    - self.variables という辞書を通じて、名前付き変数(ノード)を管理
    - 各変数は 'logic' という項目で、「どうやって value を計算するか」をツリー構造で表す
    - exec_calc() を実行すると、ツリーを再帰的に辿って value を計算し、self.variables[key]['value'] に格納
    - 遺伝的プログラミングに必要な mutation や crossover(一部)などの仕組みを提供
    """

    def __init__(
        self, 
        majorid="",
        gval_list=[],
        defined_shapes={},
        use_gval=False
    ):
        """
        コンストラクタ

        Parameters:
        ----------
        majorid : str
            系統ID (個体群の系譜を識別するために使う)
        gval_list : list of str
            グローバル変数として参照可能なキーの一覧
        defined_shapes : dict
            {'input_size': 3, 'output_size': 2} といった形で明示的に定義されたシェイプのセット
        use_gval : bool
            GVAL(グローバル変数ノード)を使用するかどうか
        """
        self.majorid = majorid  # 個体を識別するID (8文字の乱数など)
        self.variables = [{}]   # キー: 変数名, 値: 変数の辞書 (logic, shape, valueなど)
        self.score_history = [] # 評価の履歴を記録
        self.score = 0          # 平均スコアなど最終的に格納
        self.node_count = 0     # ロジック上のノード数 (複雑度を表す)
        self.fingerprint = ""   # 個体の「指紋」(重複チェック用ハッシュ)
        self.use_gval = use_gval

        # シェイプ衝突チェック (同じ名前で違う数値が割り当たっていないか)
        self.defined_shapes = defined_shapes
        values_set = set(defined_shapes.values())
        if len(values_set) != len(defined_shapes.values()):
            raise Exception("Shape collision detected. 同一のshape名に異なる値が重複。")

        # グローバル値 (GVAL) の初期化
        self.gval = {}
        for gval_item in gval_list:
            self.gval[gval_item] = 0
        self.gval_list = gval_list

        # 変数作成や変異などに関するパラメータ
        self.VAR_CREATION_RATE = 0.1   # 新規変数を作る確率
        self.MAKE_CONST_RATE = 0.3     # 定数化の確率
        self.MUTATION_STRENGTH = 10
        self.TUNING_STRENGTH = 10
        self.UNUSED_VAR_TTL = 100      # 使われないまま何世代(TTL)を超えた変数を削除

        # 派生クラスで上書きする演算マップ(FUNC_MASTER)
        self.FUNC_MASTER = {}

        # 評価レポート用のprogress (任意の6つの指標を保持)
        self.progress = {}
        for i in range(1, 7):
            self.progress[i] = 0

    def root(self, a, shape=None):
        """
        'root' 演算: 何もしないで子の値を返すだけ。
        ロジックツリーの最上位に存在する関数として使う。
        """
        return a

    def get_prog_str(self):
        """
        self.progress の値を一文字ずつ可視化して返す。
        大きい値ほど良い(S,A,B,C,.)のように変換している。

        Returns:
        --------
        str
            ex. "SABC.." のような進捗指標文字列
        """
        result_list = []
        for key in self.progress.keys():
            prog = self.progress[key]
            if prog > 0.99:
                result_list.append('S')
            elif prog > 0.8:
                result_list.append('A')
            elif prog > 0.2:
                result_list.append('B')
            elif prog > 0.05:
                result_list.append('C')
            else:
                result_list.append('.')
        return "".join(result_list)

    def set_gval(self, name, value):
        """
        GVAL(グローバル変数)の値を外部からセットする。
        """
        if name in self.gval_list:
            self.gval[name] = value
        else:
            print("GVAL " + name + " not exists.")
            exit()

    def get_gval(self, name):
        """
        GVAL(グローバル変数)の値を取得。
        """
        try:
            return self.gval[name]
        except:
            print("GVAL " + name + " not exists.")
            exit()

    def bake_logic(self, logic):
        """
        ロジックツリーを再帰的に辿り、
        logic['content'] から self.FUNC_MASTER の実関数参照を引っ張ってきて
        logic['ref'] にバインドする。

        * JSON読み込み直後は 'ref' が空なので、ここで改めて紐付けを行う。
        """
        if logic is None:
            return
        if logic['type'] == FUNC:
            logic['ref'] = self.FUNC_MASTER[logic['content']]['func']
            for arg in logic['args']:
                self.bake_logic(arg)

    def unbake_logic(self, logic):
        """
        bake_logicの逆処理。
        'ref' を削除して純粋な構造に戻す。

        * JSONで保存したり、他所へ転送するときにPython関数参照を含めたままだと困るため。
        """
        if logic is None:
            return
        if 'ref' in logic:
            logic.pop('ref', None)
            if logic['type'] == FUNC and 'args' in logic:
                for arg in logic['args']:
                    self.unbake_logic(arg)
    
    def get_code(self):
        """
        現在の self.variables を、JSON文字列に変換して返す。
        その前に unbake_logic() をして、参照を削除しておく。
        """
        for key in self.variables:
            # 一旦中身のvalueを0で埋める (実際の値は保存不要)
            self.variables[key]['value'] = np.tile(0, self.variables[key]['shape'])
            self.unbake_logic(self.variables[key]['logic'])
        return npobj2json(self.variables)

    def set_code(self, json_str):
        """
        JSON文字列を受け取り、self.variables に復元。
        その後 bake_logic() を呼んで関数参照を復元。
        さらに recalc_shape() で形状の置き換えを行う。
        """
        variables_dict = json2npobj(json_str)
        self.variables = variables_dict
        for key in variables_dict:
            if self.variables[key]['logic']:
                self.bake_logic(self.variables[key]['logic'])
        self.recalc_shape()

    def recalc_shape(self):
        """
        シェイプの再計算:
        -1 -> self.defined_shapes['input_size']
        -2 -> self.defined_shapes['output_size']

        'input' や 'output' のシェイプに合わせて、ツリーを再帰的に走査して変換する。
        """
        # 変換テーブルの作成 (例: inputのshapeが(3,)なら3->-1という一時表現)
        replace_table1 = {
            self.variables['input']['shape'][0]: -1,
            self.variables['output']['shape'][0]: -2,
        }
        # 上記の-1, -2を最終的なdefined_shapesの値に置き換えるためのテーブル
        replace_table2 = {
            -1: self.defined_shapes['input_size'],
            -2: self.defined_shapes['output_size'],
        }

        def dfs_recalc_shape(node, table):
            """
            再帰的にnode['shape']を辿り、tableにマッチする次元は置き換える
            """
            new_shape = []
            for idx in range(len(node['shape'])):
                if node['shape'][idx] in table:
                    new_shape.append(table[node['shape'][idx]])
                else:
                    new_shape.append(node['shape'][idx])
            node['shape'] = tuple(new_shape)

            # 子ノードがあるなら再帰
            if node['type'] == FUNC:
                for arg in node['args']:
                    dfs_recalc_shape(arg, table)

        # まず(-1, -2)化 → その後に実際の数字に直す という二段階
        for key in self.variables:
            var = self.variables[key]
            if var['logic']:
                dfs_recalc_shape(var['logic'], replace_table1)
                dfs_recalc_shape(var['logic'], replace_table2)

    def reset_score(self):
        """
        評価履歴をクリアし、スコアを0に戻す。
        """
        self.score_history = []
        self.score = 0

    def reset_progress(self):
        """
        progress配列をすべて0に初期化。
        """
        for key in self.progress:
            self.progress[key] = 0

    def resize_progress(self, attempts_count):
        """
        progressの値を試行回数に応じて正規化。
        (累計でカウントしているため、最終的に平均的な値にしておく)
        """
        for key in self.progress:
            self.progress[key] /= attempts_count

    def set_values(self, inputs_array):
        """
        外部から入力を受け取り、self.variables[input_key]['value'] に代入。
        ex) inputs_array = {'input': np.array([...])}
        """
        for input_key in inputs_array:
            self.variables[input_key]['value'] = inputs_array[input_key]

    def get_values(self):
        """
        すべての変数の value をまとめて返す。
        Returns:
        --------
        dict
            ex) {'input': array([...]), 'output': array([...]), ...}
        """
        result = {}
        for key in self.variables:
            result[key] = self.variables[key]['value']
        return result

    def average_score(self):
        """
        self.score_history に記録されたスコアの平均を self.score に格納。
        """
        if len(self.score_history) != 0:
            try:
                self.score = sum(self.score_history)/len(self.score_history)
            except FloatingPointError:
                print("FloatingPointError in score_history")
                self.score = 0
        else:
            self.score = 0

    def exec_calc(self):
        """
        変数のlogicを再帰的に評価して、valueを更新する処理。
        まず各変数に対し 'updated'フラグをリセットし、
        'output' の更新を呼び出すことで、必要な変数を再帰的にupdateしていく。
        """
        def update_variable(variable, var_chain):
            """
            ある変数がまだ updated でなければ、そのlogicを再帰計算して value を更新。
            """
            if variable['logic'] is not None:
                temp = dfs_exec_calc(variable['logic'], var_chain)
                variable['value'] = temp
                variable['updated'] = True

        def dfs_exec_calc(node, var_chain):
            """
            実際にロジックを深く辿って計算する本体の関数。
            """
            _content = node['content']
            if node['type'] == FUNC:
                # 子ノードを先に計算し、その結果を関数に渡す
                child_args = [dfs_exec_calc(arg, var_chain) for arg in node['args']]
                result = node['ref'](*child_args, shape=node['shape'])

                # シェイプが合っているか最終チェック
                if np.shape(result) != node['shape']:
                    print("(E) SHAPE MISMATCH!")
                    print(node)
                    raise Exception("(E) SHAPE MISMATCH!")
                return result

            elif node['type'] == CONST:
                # 定数ノード → shapeいっぱいに敷き詰めたndarrayを返す
                return np.tile(_content, node['shape'])

            elif node['type'] == VAR:
                # 変数ノード → 変数名(_content)を参照
                target_var = self.variables[_content]
                if target_var['updated']:
                    # 既に更新済みならそのまま
                    return target_var['value']
                else:
                    # 無限ループ防止: 同じ変数を2度辿るときは更新を諦めて現在のvalue
                    if _content in var_chain:
                        target_var['updated'] = True
                        return target_var['value']

                    var_chain.append(_content)
                    update_variable(target_var, var_chain)
                    return target_var['value']

            elif node['type'] == GVAL:
                # グローバル変数ノード(GVAL) → shapeぶんタイルして返す
                return np.tile(self.get_gval(_content), node['shape'])

        # 全変数の 'updated' フラグをリセット
        for key in self.variables:
            if self.variables[key].get('logic') is not None:
                self.variables[key]['updated'] = False

        # 'input' は外部から値が与えられているものなので updated = True にしておく
        self.variables['input']['updated'] = True

        # 出力変数の更新を呼び出せば、必要に応じて再帰的にすべて計算される
        update_variable(self.variables['output'], ['output'])

    def post_action(self):
        """
        進化世代ごとなどで呼ばれる後処理:
        1) ノード数(node_count)のカウント
        2) fingerprint(指紋)の計算
        3) 未使用変数のunused_countを進め、TTLを越えたら削除

        Returns:
        --------
        bool
            shape計算やロジックが破綻していないならTrue
        """
        def dfs_post_action(logic, variable_usage, counter=0, content_str=""):
            """
            ロジック木を深く探索し、ノードをカウント。
            fingerprint作成用の文字列 (typeやcontent) を連結する。

            variable_usage: dict
                変数の使用状況を記録する
            counter: int
                ノードの累計数
            content_str: str
                fingerprintを作るために連結する文字列
            """
            cs = str(logic['type']) + str(logic['content'])

            if logic['type'] == FUNC and 'args' in logic:
                for arg in logic['args']:
                    _, counter, cs = dfs_post_action(arg, variable_usage, counter=counter+1, content_str=cs)
                    if counter is None:
                        return None, None, None
                return True, counter, content_str + cs

            elif logic['type'] == CONST:
                return True, counter, content_str + cs

            elif logic['type'] == VAR:
                # 使用した変数名を記録
                variable_usage[logic['content']] = True
                return True, counter, content_str + cs

            elif logic['type'] == GVAL:
                # グローバル変数を使っていることを記録
                variable_usage[logic['content']] = True
                return True, counter, content_str + cs

        # まず全変数をbakeしておく(関数参照を準備)
        variable_usage = {}
        for key in self.variables:
            self.bake_logic(self.variables[key]['logic'])
            variable_usage[key] = False

        self.node_count = 0
        content_str = ""
        # 全変数のロジックを解析
        for key in self.variables:
            var = self.variables[key]
            if var['logic']:
                _, counter, content_str = dfs_post_action(var['logic'], variable_usage, content_str=content_str)
                if counter is None:
                    return False
                self.node_count += counter

        # fingerprint計算 (content_str をSHA256でハッシュ)
        m = hashlib.sha256()
        m.update(content_str.encode())
        self.fingerprint = m.hexdigest()

        # 未使用変数の削除チェック
        delete_variable_keys = []
        for key in self.variables:
            if key in variable_usage and variable_usage[key]:
                # 使われている
                self.variables[key]['used'] = True
                self.variables[key]['unused_count'] = 0
            else:
                # 未使用
                self.variables[key]['used'] = False
                self.variables[key]['unused_count'] += 1

                # TTLを超えたら削除
                if self.variables[key]['unused_count'] > self.UNUSED_VAR_TTL and (not self.variables[key]['fixed']):
                    delete_variable_keys.append(key)

        for key in delete_variable_keys:
            self.variables.pop(key)

        return True

    def select_random_node(self, logic, types=[FUNC, VAR, CONST, GVAL]):
        """
        ロジック木を深く探索し、指定した type (FUNC,VAR,CONST,GVALなど) のノード候補を集める。
        そこからランダムに1つを選んで (インデックス,ノード) を返す。

        Returns:
        --------
        (int, dict) or (None, None)
            ノードが見つかれば(index, node)、無ければ(None, None)
        """
        def dfs_select_random_node(node, depth=0, index=0, result=[]):
            # depth=0 (最上位) は選ばないようにしているので注意
            if depth != 0:
                result.append((index, node))
            if node['type'] == FUNC and 'args' in node:
                for arg in node['args']:
                    index += 1
                    index = dfs_select_random_node(arg, depth+1, index, result)
            return index
        
        result = []
        dfs_select_random_node(logic, result=result)
        filtered_result = [item for item in result if item[1]['type'] in types]
        if not filtered_result:
            return None, None
        chosen_pair = random.choice(filtered_result)
        return chosen_pair[0], chosen_pair[1]

    def seed_const(self):
        """
        変異などで新しい定数を生成するときの乱数を返す。
        ここでは -10 ~ 10 のintをランダムに返すだけ。
        """
        return random.randint(-10, 10)

    def common_mutation(self):
        """
        全個体共通の基本的な突然変異。
        一定確率 (VAR_CREATION_RATE) で新しい変数を作るなど。
        """
        if random.random() < self.VAR_CREATION_RATE:
            self.make_variable()

    def tuning(self):
        """
        微調整: 定数(CONST)ノードの値をランダムに変えてみる。
        """
        loop_count = random.randint(0, self.TUNING_STRENGTH)
        for _ in range(loop_count):
            keys_with_logic = [
                k for k,v in self.variables.items()
                if v['logic'] is not None and v['used']
            ]
            if not keys_with_logic:
                break
            target_key = random.choice(keys_with_logic)
            _, node = self.select_random_node(self.variables[target_key]['logic'], [CONST])
            if node:
                node['content'] = self.seed_const()

    def mutation(self):
        """
        突然変異:
        1) ノードを再構成(mutation1 or mutation2)
        2) 追加でmutation3(ノードを変数化)することも
        """
        def exec_mutation(var_name):
            # 50%でmutation1, 50%でmutation2
            if random.random() < 0.5:
                _, node = self.select_random_node(self.variables[var_name]['logic'])
                self.mutation1(node)
            else:
                _, node = self.select_random_node(self.variables[var_name]['logic'])
                self.mutation2(node)

            # さらに50%の確率でmutation3
            if random.random() < 0.5:
                index, node = self.select_random_node(self.variables[var_name]['logic'], [FUNC])
                if node:
                    self.mutation3(node, index)

        keys_with_logic = [
            k for k,v in self.variables.items()
            if v['logic'] is not None and v['used']
        ]
        if not keys_with_logic:
            return

        target_key = random.choice(keys_with_logic)
        exec_mutation(target_key)

    def dfs_mutation1(self, node, depth=0):
        """
        ノードをまるごと別の内容に再帰的に置き換える。

        depthが深い場合は定数や変数を優先し、浅い場合は関数を含むより大きなツリーを生成する可能性を上げる。
        use_gvalがTrueの場合はGVALも候補に含める。
        """
        if self.use_gval:
            choice_list = [CONST, VAR, FUNC, GVAL]
        else:
            choice_list = [CONST, VAR, FUNC]

        # 深いところでは定数/変数だけにすることが多い(大きくしすぎない工夫)
        if depth >= 2:
            if self.use_gval:
                choice_list = [CONST, VAR, GVAL]
            else:
                choice_list = [CONST, VAR]

        choice = random.choice(choice_list)

        if choice == CONST:
            # 定数ノードに置き換え
            node['type'] = CONST
            node['content'] = self.seed_const()
            node.pop('ref', None)
            node.pop('args', None)
            return

        elif choice == GVAL:
            node['type'] = GVAL
            node['content'] = random.choice(self.gval_list) if self.gval_list else 0
            node.pop('ref', None)
            node.pop('args', None)
            return

        elif choice == VAR:
            # 同じshapeの変数を1つ探す
            break_count = 0
            while True:
                selected_key = random.choice(list(self.variables))
                if self.variables[selected_key]['shape'] == node['shape']:
                    node['type'] = VAR
                    node['content'] = selected_key
                    node.pop('ref', None)
                    node.pop('args', None)
                    return
                if break_count > 20:
                    break
                break_count += 1
            # 見つからなければ定数にfallback
            node['type'] = CONST
            node['content'] = self.seed_const()
            node.pop('ref', None)
            node.pop('args', None)
            return

        elif choice == FUNC:
            # 関数ノード: shapeRefを使って子ノードのシェイプを確定 → さらに再帰的に子をmutation1
            keys_list = list(self.FUNC_MASTER)
            if 'root' in keys_list:
                keys_list.remove('root')
            input_lineups = [self.variables[k]['shape'] for k in self.variables]
            child_shapes = None
            break_count = 0
            while (child_shapes is None) and (break_count < 10) and (keys_list):
                func_name = random.choice(keys_list)
                # shapeRef(出力シェイプ, 入力候補リスト) → 子ノードのシェイプ
                child_shapes = self.FUNC_MASTER[func_name]['shapeRef'](node['shape'], input_lineups)
                break_count += 1
            if child_shapes is None:
                # 失敗 → 定数化
                node['type'] = CONST
                node['content'] = self.seed_const()
                node.pop('ref', None)
                node.pop('args', None)
                return

            # FUNCノードとして設定
            node['type'] = FUNC
            node['content'] = func_name
            node['args'] = []
            for cs in child_shapes:
                new_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                new_node = {
                    'shape': cs,
                    'id': new_id
                }
                self.dfs_mutation1(new_node, depth+1)
                node['args'].append(new_node)

    def mutation1(self, node):
        """
        mutation1: dfs_mutation1を実行し、ノードを大きく再生成する。
        """
        if node:
            self.dfs_mutation1(node)

    def mutation2(self, node):
        """
        mutation2: すでにあるノードを「別の関数」で包み込むなどの操作(関数挿入)を行う。
        """
        def dfs_mutation2(target_node_id, cur_node):
            if cur_node['type'] == FUNC and 'args' in cur_node:
                for i, arg in enumerate(cur_node['args']):
                    if arg['id'] == target_node_id:
                        keys_list = list(self.FUNC_MASTER)
                        if 'root' in keys_list:
                            keys_list.remove('root')
                        if not keys_list:
                            return False

                        func_name = random.choice(keys_list)
                        insert_position = random.randint(0, self.FUNC_MASTER[func_name]['arg_count'] - 1)
                        pinned_shape = [None]*self.FUNC_MASTER[func_name]['arg_count']
                        pinned_shape[insert_position] = arg['shape']
                        input_lineups = [self.variables[k]['shape'] for k in self.variables]
                        child_shapes = self.FUNC_MASTER[func_name]['shapeRef'](arg['shape'], input_lineups, pinned_shape=pinned_shape)

                        # arg 自体を新しいFUNCノードの中に再配置
                        arg['id'] = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                        arg['type'] = FUNC
                        arg['content'] = func_name
                        arg['args'] = [None]*self.FUNC_MASTER[func_name]['arg_count']
                        arg['args'][insert_position] = arg

                        # 他の引数スロットを埋める
                        for idx, shape_ in enumerate(child_shapes):
                            if idx == insert_position:
                                continue
                            new_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                            new_node = {'shape': shape_, 'id': new_id}
                            # ここでは mutation1 相当の操作を行って子ノードを生成
                            self.mutation1(new_node)
                            arg['args'][idx] = new_node
                        return True

                    # 再帰探索
                    if dfs_mutation2(target_node_id, arg):
                        return True
            return False

        if node and 'id' in node:
            dfs_mutation2(node['id'], node)

    def mutation3(self, node, index):
        """
        mutation3: ツリー内の指定インデックス(node)を「変数」に置き換え、
        その変数の中に旧nodeを root とする logic を持たせる操作。

        つまり 「一部のノードを変数化し、あとで再利用可能にする」 仕組み。
        """
        import copy

        def dfs_replace_object(cur_node, target_index, current_idx, replacement):
            """
            ツリーをDFSし、target_index番目のノードを replacement で置き換える。
            """
            if current_idx == target_index:
                # 目指す場所にきたら replacement を返す
                return replacement, current_idx + 1
            updated_node = copy.deepcopy(cur_node)

            # 子ノード(args)があれば再帰
            if 'args' in cur_node:
                updated_args = []
                for arg in cur_node['args']:
                    updated_arg, new_idx = dfs_replace_object(arg, target_index, current_idx, replacement)
                    updated_args.append(updated_arg)
                    current_idx = new_idx
                updated_node['args'] = updated_args

            return updated_node, current_idx

        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        # この変数に置き換える
        replacement = {
            'type': VAR,
            'content': random_string,
            'shape': node['shape']
        }
        # 置き換え実行
        _, _ = dfs_replace_object(node, index, 0, replacement)

        # 置き換え元のnodeを root として保持する変数を新規作成
        self.variables[random_string] = {
            'value': np.tile(0, node['shape']),
            'logic': {
                'type': FUNC,
                'content': 'root',
                'shape': node['shape'],
                'ref': None,
                'args': [node]
            },
            'shape': node['shape'],
            'init_policy': random.choice(['random', 'zero', 'one']),
            'fixed': False,
            'used': True,
            'unused_count': 0,
        }

    def make_variable(self):
        """
        mutationなどで新規変数を作りたいときに呼ぶ。
        現在のツリーに含まれるshapeをランダムに拾って、そのシェイプを持つノードを再帰的に生成する。
        """
        def dfs_collect_variation(nd, result=[]):
            result.append(nd['shape'])
            if nd['type'] == FUNC and 'args' in nd:
                for arg in nd['args']:
                    dfs_collect_variation(arg, result)

        variable_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        node_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

        shape_pool = []
        for key in self.variables:
            if self.variables[key]['logic']:
                dfs_collect_variation(self.variables[key]['logic'], shape_pool)
        # shape_poolが空なら (1,) としておく
        if not shape_pool:
            shape_for_new = (1,)
        else:
            shape_for_new = random.choice(shape_pool)

        # 新規ノードを作成 → dfs_mutation1 で何らかの木構造を生成
        new_node = {'shape': shape_for_new, 'id': node_id}
        self.dfs_mutation1(new_node)

        # 上記のノードを rootとする新しい変数を作成
        new_variable = {
            'value': np.tile(0, shape_for_new),
            'logic': {
                'type': FUNC,
                'content': 'root',
                'shape': shape_for_new,
                'ref': None,
                'args': [new_node]
            },
            'shape': shape_for_new,
            'init_policy': random.choice(['random', 'zero', 'one']),
            'fixed': False,
            'used': True,
            'unused_count': 0,
        }
        self.variables[variable_name] = new_variable

    def init_value(self):
        """
        変数の init_policy に従い、valueを初期化する。
        'zero'→np.zeros, 'one'→np.ones, 'random'→np.random.rand
        """
        for key in self.variables:
            variable = self.variables[key]
            if variable['init_policy'] == 'zero':
                variable['value'] = np.zeros(variable['shape'])
            elif variable['init_policy'] == 'one':
                variable['value'] = np.ones(variable['shape'])
            elif variable['init_policy'] == 'random':
                variable['value'] = np.random.rand(*variable['shape'])
