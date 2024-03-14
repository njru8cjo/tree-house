#include "modulebuilder.h"
#include "sklearnparser.h"
#include "xgboostparser.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

static std::unique_ptr<JsonParser> createSklearnParser()
{
    std::string modelJsonPath = "../data/sklearn_models/DT_1.json";
    return std::make_unique<SklearnParser>(modelJsonPath);
}

static std::unique_ptr<JsonParser> createXGboostParser()
{
    std::string modelJsonPath = "/home/chku/decisionTreeDialect/data/xgb_models/test.json";
    std::string statePath = "/home/chku/decisionTreeDialect/data/xgb_models/test.csv";
    return std::make_unique<XGBoostParser>(modelJsonPath);
}

int main()
{

    std::unique_ptr<JsonParser> parser = createXGboostParser();
    parser->constructForest();

    decisionTree::ModuleBuilder builder(parser->getDecisionForest());
    builder.buildHIRModule();
    ModuleOp module = builder.lowerToLLVMModule();

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    registerBuiltinDialectTranslation(*module->getContext());
    registerLLVMDialectTranslation(*module->getContext());

    // optLevel, sizeLevel, targetMachine
    auto optPipeline = makeOptimizingTransformer(0, 0, nullptr);
    
    ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;

    auto maybeEngine = ExecutionEngine::create(module, engineOptions);
    auto &engine = maybeEngine.get();
    //auto funcPtr = engine->lookup("predict");
    //auto myFunction = reinterpret_cast<float (*)(float*)>(funcPtr);

    return 0;
}

void dumpLLVMIRToFile(mlir::ModuleOp module)
{
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
    {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return;
    }

    llvm::errs() << *llvmModule << "\n";
}