#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Builders.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include<fstream>
#include<iostream>
#include<vector>
#include<algorithm>

using namespace mlir;

int main(int argc, char **argv) {


  DialectRegistry registry;
  registerAllDialects(registry);

  // Create an MLIR context and a module.
  MLIRContext context(registry);

  // Get a builder object to facilitate op creation.
  OpBuilder builder(&context);
  LowerToLLVMOptions options(&context);
  LLVMTypeConverter typeConverter(&context, options);
  
  ModuleOp m_module = mlir::ModuleOp::create(builder.getUnknownLoc(), llvm::StringRef("MyModule"));
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();



  // Define foo function type: () -> i32
  auto location = builder.getUnknownLoc();
  Type int32Type = builder.getI32Type();
  Type arrayType = LLVM::LLVMArrayType::get(int32Type, 5);
  Type ptrType = typeConverter.getPointerType(int32Type);
  auto functionType = builder.getFunctionType({ptrType}, int32Type);
  auto func = builder.create<mlir::func::FuncOp>(location, "foo", functionType);
  func.setPublic();
  
  auto insertPoint = builder.saveInsertionPoint();
  
  auto &entryBlock = *func.addEntryBlock(); a[2]
  builder.setInsertionPointToStart(&entryBlock);

  // Constants
  Value const0 = builder.create<mlir::arith::ConstantIntOp>(location, 0, int32Type);
  Value const1 = builder.create<mlir::arith::ConstantIntOp>(location, 1, int32Type);
  Value const2 = builder.create<mlir::arith::ConstantIntOp>(location, 2, int32Type);
  Value const5 = builder.create<mlir::arith::ConstantIntOp>(location, 5, int32Type);
  // Get input pointer
  Value input = entryBlock.getArgument(0);
  // Get pointer
  ValueRange idx = {const0, const2};
  Value ptr2 = builder.create<LLVM::GEPOp>(location, ptrType, arrayType, input, idx, true);
  Value data2 = builder.create<LLVM::LoadOp>(location, int32Type, ptr2);
  

  // Creating a condition, e.g., %arg > 0
  mlir::Value condition = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sgt, data2, const5);
  Block *tBlock = func.addBlock();
  Block *fBlock = func.addBlock();
  ValueRange nullList = {};
  builder.create<LLVM::CondBrOp>(location, condition, tBlock,  nullList, fBlock, nullList);
  builder.setInsertionPointToStart(tBlock);
  builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&context), const1);
  builder.setInsertionPointToStart(fBlock);
  builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&context), const2);

  

  // // Memref allocate and load store
  // int64_t shape = 10;
  // unsigned memorySpace = 0;
  // // Define the shape of the memref (e.g., 4x8).
  // MemRefType memrefType = MemRefType::get(shape, int32Type, {}, memorySpace);
  // // Define the optional memory space (e.g., 0 for the default memory space).
  // Value memref = builder.create<memref::AllocaOp>(builder.getUnknownLoc(), memrefType);
  // builder.create<memref::StoreOp>(builder.getUnknownLoc(), const42, memref, ValueRange(const2));
  // Value loadedValue = builder.create<memref::LoadOp>(builder.getUnknownLoc(), memref, ValueRange(const2));

  // LLVM allocate and load store
  // Type arrayType = LLVM::LLVMArrayType::get(int32Type, 10);
  // Type ptrType = typeConverter.getPointerType(int32Type);
  // IntegerAttr intAttr = builder.getI64IntegerAttr(0);
  // TypeAttr tyAttr = TypeAttr::get(int32Type);

  // Type elementType = mlir::IntegerType::get(&context, 32);  // Example: 32-bit integer
  // Type pointerType = typeConverter.getPointerType(arrayType);
  
  // Value arraySize = builder.create<LLVM::ConstantOp>(builder.getUnknownLoc(), int32Type, 10);
  // //::mlir::Type res, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::TypeAttr elem_type, /*optional*/bool inalloca = false);
  // //Value arr = builder.create<LLVM::AllocaOp>(location, ptrType, arrayType, arraySize, 16);
  // ValueRange idx = {const0, const2};
  // Value ptr2 = builder.create<LLVM::GEPOp>(location, ptrType, arrayType, arr, idx, true);
  // builder.create<LLVM::StoreOp>(location, constx, ptr2, 0);
  // Value loadData = builder.create<LLVM::LoadOp>(location, int32Type, ptr2);
  

  // builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&context), const2);

  m_module.push_back(func);
  m_module->dump();

  // Transfer to LLVM dialect
  mlir::PassManager pm(&context);
  mlir::ConversionTarget target(context);


  RewritePatternSet patterns(&context);
    
  // Setup target to LLVM Dialect
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  
  // Add conversion passes
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  if (failed(applyPartialConversion(m_module, target, std::move(patterns)))) {
    //signalPassFailure();
    llvm::errs() << "Decision forest lowering pass failed\n";
  }

  m_module->dump();
    
  // Print the emitted LLVM IR


  std::cout << "pass" << std::endl;
  return 0;
}

