//===- TreeTypes.cpp - Tree dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tree/TreeTypes.h"

#include "Tree/TreeDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tree;

#define GET_TYPEDEF_CLASSES
#include "Tree/TreeOpsTypes.cpp.inc"

void TreeDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Tree/TreeOpsTypes.cpp.inc"
      >();
}
