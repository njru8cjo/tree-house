
#include <string>
#include "test/maintest.h"

using namespace Treehierarchy;

bool RunXGBoostBenchmarksIfNeeded(int argc, char *argv[]) {
    for (int32_t i = 0; i < argc; i++)
        if (std::string(argv[i]).find(std::string("--xgboostBench")) != std::string::npos) {
            test::RunXGBoostNonOptimizeTests();
            test::RunXGBoostSwapOptimizeTests();
            test::RunXGBoostFlintOptimizeTests();
            return true;
        }
    return false;
}

bool RunCorrectnessTestIfNeeded(int argc, char *argv[]) {
    for (int32_t i = 0; i < argc; i++)
        if (std::string(argv[i]).find(std::string("--correctness")) != std::string::npos) {
            test::RunXGBoostCorrectnessTests();
            return true;
        }
    return false;
}

int main(int argc, char *argv[])
{
    if (RunCorrectnessTestIfNeeded(argc, argv))
        return 0;
    else if (RunXGBoostBenchmarksIfNeeded(argc, argv))
        return 0;

    return -1;
}

// void dumpLLVMIRToFile(mlir::ModuleOp module)
// {
//     llvm::LLVMContext llvmContext;
//     auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
//     if (!llvmModule)
//     {
//         llvm::errs() << "Failed to emit LLVM IR\n";
//         return;
//     }
//
//     llvm::errs() << *llvmModule << "\n";
// }
