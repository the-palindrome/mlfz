import importlib


def test_runtime_imports():
    modules = [
        "mlfz",
        "mlfz.classical",
        "mlfz.classical.knn",
        "mlfz.classical.tree.cart",
    ]

    for module in modules:
        importlib.import_module(module)
