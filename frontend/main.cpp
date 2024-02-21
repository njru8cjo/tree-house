#include "modulebuilder.h"
#include "sklearnparser.h"
#include "xgboostparser.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

using namespace mlir;

int main() {

    std::string modelJsonPath = "/home/chku/decisionTreeDialect/data/xgb_models/test.json";
    XGBoostParser parser = XGBoostParser(modelJsonPath);
    parser.constructForest();

    decisionTree::ModuleBuilder builder(parser.getDecisionForest());
    auto module = builder.buildHIRModule();
    module = builder.lowerToLLVMModule();
    
    module->dump();

    return 0;
}

void sklearnParser() {
    std::string modelJsonPath = "../data/sklearn_models/DT_1.json";
    SklearnParser parser = SklearnParser(modelJsonPath);
    parser.constructForest();
}
