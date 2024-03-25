#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"


namespace Treehierarchy
{
    namespace test
    {
        void DumpLLVMIRToFile(mlir::ModuleOp module, const std::string& filename) {
            // Register the translation to LLVM IR with the MLIR context.
            registerBuiltinDialectTranslation(*module->getContext());
            registerLLVMDialectTranslation(*module->getContext());

            // Convert the module to LLVM IR in a new LLVM IR context.
            llvm::LLVMContext llvmContext;
            auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
            if (!llvmModule) {
                llvm::errs() << "Failed to emit LLVM IR\n";
            }

            // Initialize LLVM targets.
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();

            // Configure the LLVM Module
            auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
            if (!tmBuilderOrError) {
                llvm::errs() << "Could not create JITTargetMachineBuilder\n";
            }

            auto tmOrError = tmBuilderOrError->createTargetMachine();
            if (!tmOrError) {
                llvm::errs() << "Could not create TargetMachine\n";
            }
            mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), tmOrError.get().get());

            /// Optionally run an optimization pipeline over the llvm module.
            auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
            if (auto err = optPipeline(llvmModule.get())) {
                llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
            }
            
            std::error_code ec;
            llvm::raw_fd_ostream filestream(filename, ec);
            filestream << *llvmModule;
        }
    }
}