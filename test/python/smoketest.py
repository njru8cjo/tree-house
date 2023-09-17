# RUN: %python %s | FileCheck %s

from mlir_tree.ir import *
from mlir_tree.dialects import builtin as builtin_d, tree as tree_d

with Context():
    tree_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = tree.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: tree.foo %[[C]] : i32
    print(str(module))
