# Copyright (c) 2022 Katori lab. All Rights Reserved

from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

#__init__以外の拡張子が.pyのモジュール全て宣言を行う。