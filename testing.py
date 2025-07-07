python - <<'PY'
import inspect, importlib.util, pathlib
m = importlib.util.module_from_spec(importlib.util.spec_from_file_location('mod', 'attention_flow.py'))
m.__spec__.loader.exec_module(m)
print([n for n, o in inspect.getmembers(m, inspect.isclass) if 'Scene' in [b.__name__ for b in o.__bases__]])
PY
