#!/usr/bin/env python3
"""Create separate threshold-clipped SNN4 experiment files.

Run from ~/rcrl_k with: python create_snn4_clip_variant.py
The original core, build script, model wrapper, and plot script are never edited.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CORE_DIR = ROOT / "snn4tanaka"
MODELS = ROOT / "models"

src_core = CORE_DIR / "simu.py"
src_model = MODELS / "snn_reservoir_snn4.py"
src_plot = ROOT / "test_plot_snn4_02ms.py"

new_core = CORE_DIR / "simu_clip.py"
new_setup = CORE_DIR / "setup_clip.py"
new_model = MODELS / "snn_reservoir_snn4_clip.py"
new_plot = ROOT / "test_plot_snn4_02ms_clip.py"

# Check everything BEFORE creating any new file.
for path in (src_core, src_model, src_plot):
    if not path.is_file():
        raise FileNotFoundError(f"Required source not found: {path}")
for path in (new_core, new_setup, new_model, new_plot):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

core_text = src_core.read_text(encoding="utf-8")
# This exact French LIF Euler update is intentionally the only modified statement.
pattern = r"(?m)^(?P<indent>[ \t]*)self\._V\[i\][ \t]*\+=[ \t]*\([^\n]*\)[ \t]*\*[ \t]*self\._smem\[i\][ \t]*$"
matches = list(re.finditer(pattern, core_text))
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one LIF voltage update, found {len(matches)}. No files written.")
m = matches[0]
indent = m.group("indent")
clip_block = (
    f"\n{indent}# Experimental variant: cap suprathreshold Euler overshoot."
    f"\n{indent}if self._V[i] > self._Vthre[i]:"
    f"\n{indent}    self._V[i] = self._Vthre[i]"
)
new_core_text = core_text[:m.end()] + clip_block + core_text[m.end():]

# Prefer matching either of the usual import styles, without modifying the original wrapper.
model_text = src_model.read_text(encoding="utf-8")
import_pattern = r"(?m)^(?P<prefix>[ \t]*from[ \t]+)(?P<module>snn4tanaka\.simu|simu)(?P<suffix>[ \t]+import[ \t]+[^\n]*\bCulturedSNNCore\b[^\n]*)$"
import_matches = list(re.finditer(import_pattern, model_text))
if len(import_matches) != 1:
    raise RuntimeError(
        "Could not uniquely identify 'from ...simu import CulturedSNNCore' "
        "in models/snn_reservoir_snn4.py. No files written. "
        "Show its import section to adjust this helper."
    )
im = import_matches[0]
module = im.group("module")
new_module = module.rsplit("simu", 1)[0] + "simu_clip"
new_model_text = (model_text[:im.start("module")] + new_module + model_text[im.end("module"):])

plot_text = src_plot.read_text(encoding="utf-8")
old = 'c.rc_module = "models.snn_reservoir_snn4"'
new = 'c.rc_module = "models.snn_reservoir_snn4_clip"'
if plot_text.count(old) != 1:
    raise RuntimeError(
        "Could not uniquely identify c.rc_module in the existing plot script. "
        "No files written."
    )
new_plot_text = plot_text.replace(old, new, 1)

new_setup_text = '''#!/usr/bin/env python3
# Compile ONLY the separate snn4tanaka/simu_clip.py module.
# Never builds or overwrites the original simu module.
from Cython.Distutils import build_ext
from setuptools import setup, Extension
from numpy import get_include
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
Cython.Compiler.Options.cimport_from_pyx = True

ext_modules = [
    Extension(
        "simu_clip", sources=["simu_clip.py"], language="c",
        extra_compile_args=["-O3"], extra_link_args=["-O3"],
    ),
]

setup(
    name="snn4tanaka_simu_clip",
    ext_modules=ext_modules,
    include_dirs=[get_include()],
    cmdclass={"build_ext": build_ext},
)
'''

# Every validation has succeeded. Write separate files with exclusive creation.
files = (
    (new_core, new_core_text),
    (new_setup, new_setup_text),
    (new_model, new_model_text),
    (new_plot, new_plot_text),
)
for path, contents in files:
    with path.open("x", encoding="utf-8") as f:
        f.write(contents)
    print(f"CREATED: {path.relative_to(ROOT)}")

print("\nOriginal simu.py / setup.py / model / plot files were NOT modified.")
print("Build: (cd snn4tanaka && python setup_clip.py build_ext --inplace)")
print("Test:  python test_plot_snn4_02ms_clip.py --episodes 1")
print("Control: python test_plot_snn4_02ms.py --episodes 1")
