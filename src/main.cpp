
#include <string>
#include "test/maintest.h"

using namespace Treehierarchy;

bool RunXGBoostBenchmarksIfNeeded(int argc, char *argv[]) {
    for (int32_t i = 0; i < argc; i++)
        if (std::string(argv[i]).find(std::string("--xgboostBench")) != std::string::npos) {
            test::RunXGBoostNonOptimizeTests();
            test::RunXGBoostSwapOptimizeTests();
            test::RunXGBoostFlintOptimizeTests();
            test::RunXGBoostOptimizeTests();
            return true;
        }
    return false;
}

bool RunSklearnBenchmarksIfNeeded(int argc, char *argv[]) {

    for (int32_t i = 0; i < argc; i++)
        if (std::string(argv[i]).find(std::string("--sklearnBench")) != std::string::npos) {
            test::RunSKlearnOptimizeTests();
            test::RunSKlearnFlintOptimizeTests();
            test::RunSKlearnSwapOptimizeTests();
            test::RunSKlearnNonOptimizeTests();
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

bool DumpLLVMIRIfNeeded(int argc, char *argv[]) {
    for (int32_t i = 0; i < argc; i++)
        if (std::string(argv[i]).find(std::string("--dump")) != std::string::npos) {
            test::DumpXGBoostLLVMIR();        
            test::DumpSKlearnLLVMIR();
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
    else if (RunSklearnBenchmarksIfNeeded(argc, argv))
        return 0;
    else if (DumpLLVMIRIfNeeded(argc, argv))
        return 0;

    return -1;
}

