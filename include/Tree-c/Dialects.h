//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TREE_C_DIALECTS_H
#define TREE_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Tree, tree);

#ifdef __cplusplus
}
#endif

#endif // TREE_C_DIALECTS_H
