//===- tree-plugin.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "Tree/TreeDialect.h"
#include "Tree/TreePasses.h"

using namespace mlir;

/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Tree", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<mlir::tree::TreeDialect>();
            mlir::tree::registerPasses();
          }};
}

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TreePasses", LLVM_VERSION_STRING,
          []() { mlir::tree::registerPasses(); }};
}
