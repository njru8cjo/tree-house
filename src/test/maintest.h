#include <string>
#include "mlir/IR/BuiltinOps.h"

namespace Treehierarchy
{
    namespace test
    {
        void RunXGBoostNonOptimizeTests();
        void RunXGBoostSwapOptimizeTests();
        void RunXGBoostFlintOptimizeTests();
        void RunXGBoostOptimizeTests();
        void RunXGBoostCorrectnessTests();
        void DumpXGBoostLLVMIR();

        void RunSKlearnNonOptimizeTests();
        void RunSKlearnSwapOptimizeTests();
        void RunSKlearnFlintOptimizeTests();
        void RunSKlearnOptimizeTests();
        void RunSKlearnCorrectnessTests();
        void DumpSKlearnLLVMIR();

        void DumpLLVMIRToFile(mlir::ModuleOp module, const std::string& filename);
    }
}