# Copyright (c) 2018-2023 Katori lab. All Rights Reserved
# 設定ファイルの保存・読込、モデルの生成・保存・読込に関わる関数

import os
import argparse
import numpy as np
import pandas
import datetime 
import csv
import pickle
import importlib
import logging
log = logging.getLogger('common_configurator')
log.setLevel(logging.INFO)# DEBUG, INFO, WARNING, ERROR, CRITICAL

SAVED_MODELS = "./saved_models/"
SAVED_FIGURES = "./saved_figures/"
SAVED_CONFIG = "./saved_config/"

def get_arguments():
    parser = argparse.ArgumentParser(description='A program that requires a configuration file as an argument.')
    parser.add_argument("-c","-config", type=str,dest="config", help="Path to the configuration file.")
    args = parser.parse_args()
    return args

def convert_path_to_module(path_or_module):
    # すでにドット形式であれば、そのまま返す
    if '.' in path_or_module and '/' not in path_or_module:
        return path_or_module

    # 拡張子を取り除く
    path_without_extension = path_or_module.rsplit('.', 1)[0]
    # スラッシュをドットに変換
    module_name = path_without_extension.replace('/', '.')
    return module_name

def load_config(a,class_="Config"):
    """
    pickle形式または.py形式のconfigを読み込む.
    引数：
    a:argparseでの解析結果、またはモジュール名
    """
    #print(type(a))
    if hasattr(a, 'config'):# aをargparseの結果と解釈する
        if a.config.endswith('.pkl'):
            c = pandas.read_pickle(a.config)
        else:
            #c = importlib.import_module(a.config).Config()
            module_name = convert_path_to_module(a.config)
            module = importlib.import_module(module_name)
            c = getattr(module, class_)()
    else:# aをモジュール名と解釈する
        module = importlib.import_module(a)
        c = getattr(module, class_)()

    return c

def save_config(c):
    """
    configをcsv(またはpickle)で保存する．
    """
    prepare_directory("./data/")

    ### csv 
    if hasattr(c, 'csv') and c.csv is not None:
        vx = [getattr(c,col) for col in c.columns]
        with open(c.csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(vx)
    
    ### pickle
    if hasattr(c,'pickle') and c.pickle != None:
        pd.to_pickle(c,c.pickle)

def generate_instance(c=None,module=None,class_=None):
    # Try to import the module and get the class
    print(f"generating module: {module} class: {class_}")
    try:
        module = importlib.import_module(module)
        class_ = getattr(module, class_)
    except ImportError:
        raise ImportError(f"Failed to import the specified module: {module}")
    except AttributeError:
        raise AttributeError(f"The specified class {class_} was not found in the module {module}")

    # Create and return an instance of the class
    if c is None:
        instance = class_()
    else:
        instance = class_(c)

    return instance

def generate_model(c=None,module=None,class_=None):
    
    """
    指定されたモジュールとクラス名に基づいて、クラスのインスタンスを生成・初期化して返します。

    Parameters:
        c (object, optional): モデルの設定情報を含むオブジェクト。
        `model_module` および `model_class` 属性が必要です。
        module (str, optional): インスタンスを生成するクラスが定義されているモジュール名。
        class_ (str, optional): インスタンスを生成するクラス名。

    Returns:
        object: 生成されたクラスのインスタンス。

    Raises:
        ValueError: モジュールまたはクラスが指定されていない場合。
        ImportError: 指定されたモジュールがインポートできない場合。
        AttributeError: 指定されたクラスがモジュール内に見つからない場合。

    Notes:
        `c` パラメーターには、モデルのモジュール名とクラス名を提供することもできます。
        この場合、`module` および `class_` パラメーターが指定されていない限り、それらが使用されます。
    """
    if hasattr(c, 'model_module') and hasattr(c, 'model_class'):
        model_module = c.model_module
        model_class_ = c.model_class

    if module is not None:
        model_module = module
    if class_ is not None:
        model_class_ = class_

     # Check that we have a module and class
    if model_module is None:
        raise ValueError("No module specified.")
    if model_class_ is None:
        raise ValueError("No class specified.")

    # Try to import the module and get the class
    print(f"module: {model_module} class: {model_class_}")
    try:
        module = importlib.import_module(model_module)
        class_ = getattr(module, model_class_)
    except ImportError:
        raise ImportError(f"Failed to import the specified module: {model_module}")
    except AttributeError:
        raise AttributeError(f"The specified class {model_class_} was not found in the module {model_module}")

    # Create and return an instance of the class
    if c is None:
        instance = class_()
    else:
        instance = class_(c)

    return instance


def save_model(model, filename=None, prefix=None):
    """
    モデル（クラスのインスタンス）をPickle形式で保存します。

    Parameters:
        model (object): 保存するモデルのインスタンス。
        filename (str, optional): 保存するファイルの名前。Noneの場合、現在の日時に基づく名前が使用されます。
        prefix (str, optional): 保存するファイル名の接頭辞。指定された場合、ファイル名に追加されます。
    """
    if filename is None:
        filename = string_now() + ".pkl"
        
    if prefix is not None:
        filename = prefix + string_now() + ".pkl"
    
    prepare_directory(SAVED_MODELS)

    with open(SAVED_MODELS + filename, mode='wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Model {filename} is saved. ")

def load_model(filename):
    """
    モデルをPickle形式から読み込みます。

    Parameters:
        filename (str): 読み込むファイルの名前。

    Returns:
        object: 読み込まれたモデルのインスタンス。

    Raises:
        FileNotFoundError: 指定されたファイルが見つからない場合。
        Exception: その他のエラーが発生した場合。
    """
    try:
        with open(SAVED_MODELS + filename, mode='rb') as f:
            model = pickle.load(f)
            log.info(f"Model {filename} is loaded. ")
        return model
    except FileNotFoundError:
        print(f"File {SAVED_MODELS + filename} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def prepare_directory(path):
    """
    # 指定されたディレクトリがなければ作る
    """
    if os.path.isdir(path): # ディレクトリの存在の確認
        #print("Exist Directory: %s" % (path))
        pass
    else:
        print("Create Directory: %s" % (path))
        os.makedirs(path) # ディレクトリの作成

def string_now():
    t1=datetime.datetime.now()
    s=t1.strftime('%Y%m%d_%H%M%S')
    return s

