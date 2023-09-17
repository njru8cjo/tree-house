// RUN: mlir-opt %s --load-pass-plugin=%tree_libs/TreePlugin%shlibext --pass-pipeline="builtin.module(tree-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
