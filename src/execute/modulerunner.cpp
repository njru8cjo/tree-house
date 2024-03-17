
#include "modulerunner.h"
#include <iostream>

namespace Treehierarchy
{
    namespace Execute
    {
        ModuleRunner::ModuleRunner(mlir::ModuleOp module) : m_maybeEngine(CreateExecutionEngine(module)),
                                                            m_engine(m_maybeEngine.get()), m_module(module)
        {
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

        float ModuleRunner::runInference(float *input)
        {
            auto expectedFPtr = m_engine->lookup("predict");
            if (expectedFPtr)
            {
                auto fptr = *expectedFPtr;
                typedef float (*InferenceFunc_t)(float *);
                auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(fptr);

                return inferenceFuncPtr(input);
            }

            return -1;
        }
    }
}