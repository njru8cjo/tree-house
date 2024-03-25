#include <iostream>
#include "modulerunner.h"

namespace Treehierarchy
{
    namespace Execute
    {
        ModuleRunner::ModuleRunner(mlir::ModuleOp module) : m_maybeEngine(CreateExecutionEngine(module)),
                                                            m_engine(m_maybeEngine.get()), m_module(module)
        {
            m_inferenceFuncPtr = GetFunctionAddress("predict");
        }

        llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> ModuleRunner::CreateExecutionEngine(mlir::ModuleOp module)
        {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();

            mlir::registerLLVMDialectTranslation(*module->getContext());
            mlir::registerBuiltinDialectTranslation(*module->getContext());

            // optLevel, sizeLevel, targetMachine
            auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);

            mlir::ExecutionEngineOptions engineOptions;
            engineOptions.transformer = optPipeline;

            auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
            assert(maybeEngine);

            return maybeEngine;
        }

        void ModuleRunner::runInference(float *input, float *result)
        {
            typedef float (*InferenceFunc_t)(float *, float*);
            auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(m_inferenceFuncPtr);

            inferenceFuncPtr(input, result);
        }

        void *ModuleRunner::GetFunctionAddress(const std::string& functionName) {
            auto expectedFptr = m_engine->lookup(functionName);
            if (!expectedFptr)
                return nullptr;
            auto fptr = *expectedFptr;
            return fptr;
        }
    }
}