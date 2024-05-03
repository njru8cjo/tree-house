#include <string>
#include <iostream>
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

        inline bool FPEqual(float a, float b, float epsilon)
        {
            bool ret = std::abs(a - b) < epsilon;

            if (!ret) {
                std::cout << a << " != " << b << std::endl;
                std::cout << (a - b) << std::endl;
                std::cout << "x in hex: " << std::hexfloat << a << std::endl;
                std::cout << "y in hex: " << std::hexfloat << b << std::endl;
            }
            return ret;
        }
    }
}