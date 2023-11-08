//===- tree-opt.cpp ---------------------------------------*- C++ -*-===//
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "Tree/TreeDialect.h"
#include "Tree/TreePasses.h"

#include "json.hpp"
#include<fstream>
#include<iostream>
#include<vector>
#include<algorithm>
using json = nlohmann::json;

struct node{
int id;
int feature;
float threshold;
node* leftChild;
node* rightChild;
int prediction;
};

node* buildTree(json jstr) {
  node* res = new node();
  res->id = jstr["id"];
  if(jstr["isCategorical"] == "False"){ // Split node
    res->feature = jstr["feature"];
    res->threshold = jstr["split"];
    res->leftChild = buildTree(jstr["leftChild"]);
    res->rightChild = buildTree(jstr["rightChild"]);
    res->prediction = -1;
  }
  else { // Leaf node
    std::vector<float> vec = jstr["prediction"].get<std::vector<float>>();
    auto maxIt = std::max_element(vec.begin(), vec.end());
    res->prediction = std::distance(vec.begin(), maxIt);
  }
  return res;
}

using namespace mlir;

int main(int argc, char **argv) {

  // Read JSON file

  std::ifstream f("../data/DT_5.json");
  json json_obj;
  f >> json_obj;
  
  // TODO: Build tree
  node* root = buildTree(json_obj[0]);

  // // TODO: Lower to tree dialect
  // DialectRegistry registry;
  // registerAllDialects(registry);
  // registry.insert<func::FuncDialect>();

  // // Create an MLIR context and a module.
  // MLIRContext context(registry);

  // // Get a builder object to facilitate op creation.
  //OpBuilder builder(&context);
  
  // ModuleOp m_module = mlir::ModuleOp::create(builder.getUnknownLoc(), llvm::StringRef("MyModule"));
  // context.getOrLoadDialect<func::FuncDialect>();
  // context.getOrLoadDialect<scf::SCFDialect>();
  // context.getOrLoadDialect<arith::ArithDialect>();
  // context.getOrLoadDialect<LLVM::LLVMDialect>();



  // // Define a function type: (i32) -> i32
  // auto location = builder.getUnknownLoc();
  // auto int32Type = builder.getI32Type();
  // auto functionType = builder.getFunctionType(int32Type, int32Type);
  // auto func = builder.create<mlir::func::FuncOp>(location, "simple", functionType);
  // func.setPublic();
  
  // auto insertPoint = builder.saveInsertionPoint();
  
  // auto &entryBlock = *func.addEntryBlock();
  // builder.setInsertionPointToStart(&entryBlock);
  // auto constVal = builder.create<mlir::arith::ConstantIntOp>(location, 16, int32Type);
  

  // // Creating a condition, e.g., %arg > 0
  // mlir::Value arg = func.getArgument(0);
  // mlir::Value condition = builder.create<mlir::arith::CmpIOp>(
  // UnknownLoc::get(&context), mlir::arith::CmpIPredicate::sgt, arg, builder.create<mlir::arith::ConstantIntOp>(location, 0, int32Type));

  // // Create 'then' and 'else' blocks
  // mlir::Block *thenBlock = func.addBlock();
  // mlir::Block *elseBlock = func.addBlock();
  
  // // Set insertion points and fill in the 'then' and 'else' regions
  // builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  // builder.create<mlir::LLVM::ReturnOp>(mlir::UnknownLoc::get(&context), static_cast<Value>(const_1));

  // builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  // builder.create<mlir::LLVM::ReturnOp>(mlir::UnknownLoc::get(&context), static_cast<Value>(const_2));



  // m_module.push_back(func);
  
  // // // Print out the module.
  // m_module->dump();



  // // LLVM
  // mlir::PassManager pm(&context);
  // mlir::ConversionTarget target(context);
    
  // // Setup target to LLVM Dialect
  // target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  // target.addIllegalDialect<mlir::scf::SCFDialect>();
  
  // // Add conversion passes
  // pm.addPass(mlir::createConvertToLLVMPass());
    
  // // Run the pass manager
  // if (mlir::failed(pm.run(m_module)))
  //     return -1;

  // // Translate to LLVM IR
  // std::string llvmIR;
  // llvm::LLVMContext llvmContext;
  // auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  // if (!llvmModule) {
  //   llvm::errs() << "Failed to emit LLVM IR\n";
  //   return -1;
  // }
    
  // // Print the emitted LLVM IR


  std::cout << "pass" << std::endl;
  return 0;
}

