//===- TreeTypes.td - Tree dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TREE_TYPES
#define TREE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Tree/TreeDialect.td"

//===----------------------------------------------------------------------===//
// Tree type definitions
//===----------------------------------------------------------------------===//

class Tree_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tree_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Tree_node : Tree_Type<"Node", "node"> {
    let summary = "Tree custom type";
    let description = "Custom type in tree dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // TREE_TYPES
