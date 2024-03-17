#include "test/maintest.h"
//#include "test/correctnesstest.h"

using namespace Treehierarchy;

int main()
{
    // test::RunXGBoostNonOptimizeTests();
    //test::RunXGBoostSwapOptimizeTests();
    // test::RunXGBoostFlintOptimizeTests();
    test::RunXGBoostCorrectnessTests();


    return 0;
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
