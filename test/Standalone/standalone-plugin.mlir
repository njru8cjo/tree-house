// RUN: mlir-opt %s --load-dialect-plugin=%tree_libs/TreePlugin%shlibext --pass-pipeline="builtin.module(tree-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @tree_types(%arg0: !tree.custom<"10">)
  func.func @tree_types(%arg0: !tree.custom<"10">) {
    return
  }
}
