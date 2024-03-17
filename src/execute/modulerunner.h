#ifndef MODULE_RUNNER_H
#define MODULE_RUNNER_H

#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace Treehierarchy
{
    namespace Execute
    {
        class ModuleRunner
        {

        protected:
            llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> m_maybeEngine;
            std::unique_ptr<mlir::ExecutionEngine> &m_engine;
            mlir::ModuleOp m_module;

            llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> CreateExecutionEngine(mlir::ModuleOp module);

        public:
            ModuleRunner(mlir::ModuleOp module);
            float runInference(float *input);
        };
    }
}

#endif