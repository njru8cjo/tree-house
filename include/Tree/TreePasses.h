//===- TreePasses.h - Tree passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef TREE_TREEPASSES_H
#define TREE_TREEPASSES_H

#include "Tree/TreeDialect.h"
#include "Tree/TreeOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tree {
#define GEN_PASS_DECL
#include "Tree/TreePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Tree/TreePasses.h.inc"
} // namespace tree
} // namespace mlir

#endif
