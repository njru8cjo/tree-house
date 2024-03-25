#include <libgen.h>
#include <unistd.h>

#include "csvreader.h"
#include "maintest.h"
#include "modulerunner.h"
#include "sklearnparser.h"

namespace Treehierarchy
{
    namespace test
    {
        const std::string modelNames[] = {"adult", "bank", "letter", "magic", "satlog", "sensorless", "spambase", "wine-quality"};
        const int32_t NUM_RUNS = 10000;

        static std::string GetRepoPath()
        {
            char exePath[PATH_MAX];
            memset(exePath, 0, sizeof(exePath));
            if (readlink("/proc/self/exe", exePath, PATH_MAX) == -1)
                return std::string("");
            char *execDir = dirname(exePath);
            char *buildDir = dirname(execDir);
            char *repoPath = dirname(buildDir);
            return repoPath;
        }

        static double RunSklearnTest(JsonParser &parser, std::string testCsvPath) {
            
            parser.ConstructForest();
            ModuleOp module = parser.buildHIRModule();
            module = parser.lowerToLLVMModule();

            // module->dump();

            Execute::ModuleRunner runner(module);

            utils::CSVReader reader(testCsvPath);
            std::vector<std::vector<float>> inputData;

            for (size_t i = 0; i < reader.GetRowNum(); i++)
            {
                auto row = reader.GetRowOfType<float>(i);
                inputData.push_back(row);
            }

            
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            for (int32_t trial = 0; trial < NUM_RUNS; trial++)
            {
                for (auto &input : inputData)
                {
                    std::vector<float> result(parser.getForestClassNum(), 0);
                    runner.runInference(input.data(), result.data());
                }
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            int64_t timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            int64_t numSamples = NUM_RUNS * inputData.size();
            auto timePerSample = (double)timeTaken / (double)numSamples;
            return timePerSample;
        }

        void RunSKlearnNonOptimizeTests()
        {
            std::puts("Running Without Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n\n";
            }           
        }

        void RunSKlearnSwapOptimizeTests()
        {
            std::puts("Running Swap Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_swap = true;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n\n";
            }           
        }

        void RunSKlearnFlintOptimizeTests()
        {
            std::puts("Running Flint Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_flint = true;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n\n";
            }           
        }

        void RunSKlearnOptimizeTests()
        {
            std::puts("Running Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_swap = true;
                option.enable_ra = true;
                option.enable_flint = true;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n";
            }
            std::cout << std::endl;
        }

        void DumpSKlearnLLVMIR()
        {
            std::puts("Dumping SKlearn LLVMIR...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto dumpFileName =  GetRepoPath() + "/skll/" + modelName + ".ll";

                BuildOptions option;
                option.enable_swap = true;
                option.enable_ra = true;
                option.enable_flint = true;
                SklearnParser parser(modelJsonPath, option);
                parser.ConstructForest();
                ModuleOp module = parser.buildHIRModule();
                module = parser.lowerToLLVMModule();

                DumpLLVMIRToFile(module, dumpFileName);
            }
        }
    }
}