//===- TreeDialect.cpp - Tree dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tree/TreeDialect.h"
#include "Tree/TreeOps.h"
#include "Tree/TreeTypes.h"

using namespace mlir;
using namespace mlir::tree;

#include "Tree/TreeOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tree dialect.
//===----------------------------------------------------------------------===//

void TreeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tree/TreeOps.cpp.inc"
      >();
  registerTypes();
}
