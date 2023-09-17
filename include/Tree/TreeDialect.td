//===- TreeDialect.td - Tree dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TREE_DIALECT
#define TREE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tree dialect definition.
//===----------------------------------------------------------------------===//

def Tree_Dialect : Dialect {
    let name = "tree";
    let summary = "A tree out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::tree";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base tree operation definition.
//===----------------------------------------------------------------------===//

class Tree_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tree_Dialect, mnemonic, traits>;

#endif // TREE_DIALECT
