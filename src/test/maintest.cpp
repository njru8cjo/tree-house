#include <iostream>
#include <unistd.h>
#include <libgen.h>

#include "sklearnparser.h"
#include "xgboostparser.h"
#include "modulerunner.h"

using namespace mlir;

namespace Treehierarchy
{
    namespace test
    {
        inline bool FPEqual(float a, float b) {
            const float scaledThreshold = std::max(std::fabs(a), std::fabs(b))/1e8;
            const float threshold = std::max(float(1e-6), scaledThreshold);
            auto sqDiff = (a-b) * (a-b);
            bool ret = sqDiff < threshold;
            if (!ret)
                std::cout << a << " != " << b << std::endl;
            return ret;
        }

        const std::string modelNames[] = {"abalone", "airline", "airline-ohe", "covtype", "epsilon", "letters", "higgs", "year_prediction_msd"};
        const int32_t NUM_RUNS = 20;

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

        static void verifyXGBoostResult(JsonParser &parser, std::string testCsvPath, std::string answerCsvPath)
        {
            parser.ConstructForest();

            ModuleOp module = parser.buildHIRModule();
            module = parser.lowerToLLVMModule();


            Execute::ModuleRunner runner(module);

            utils::CSVReader testCaseReader(testCsvPath);
            std::vector<std::vector<float>> inputData;

            for (size_t i = 0; i < testCaseReader.GetRowNum(); i++)
            {
                auto row = testCaseReader.GetRowOfType<float>(i);
                inputData.push_back(row);
            }

            utils::CSVReader answerReader(answerCsvPath);
            auto answers = answerReader.GetRowOfType<float>(0);

            int i = 0;
            for (auto &input : inputData) // 2000
            {
                auto result = runner.runInference(input.data());                
                assert(FPEqual(answers[i], result));
                i++;
            }
        }

        static double RunXGBoostTest(JsonParser &parser, std::string testCsvPath)
        {
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
            for (int32_t trial = 0; trial < NUM_RUNS; trial++) // 500
            {
                for (auto &input : inputData) // 2000
                {
                    runner.runInference(input.data());
                }
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            int64_t timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            int64_t numSamples = NUM_RUNS * inputData.size();
            auto timePerSample = (double)timeTaken / (double)numSamples;

            // std::cout << "sample nums: " << numSamples << " time take: " << timeTaken << "\n";

            return timePerSample;
        }

        void RunXGBoostNonOptimizeTests()
        {
            std::puts("========================No Optimize========================\n");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                XGBoostParser parser(modelJsonPath, option);

                std::cout << modelName << "time consuming: " << RunXGBoostTest(parser, testCsvPath) << "\n";
            }
        }

        void RunXGBoostSwapOptimizeTests()
        {
            std::puts("========================Swap Optimize========================\n");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto stateCsvPath = testModelsDir + "/" + modelName + ".prob.csv";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                XGBoostParser parser(modelJsonPath, option, stateCsvPath);
                std::cout << "time consuming: " << RunXGBoostTest(parser, testCsvPath) << "\n";
            }
        }

        void RunXGBoostFlintOptimizeTests()
        {
            std::puts("========================Flint Optimize========================\n");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_flint = true;
                XGBoostParser parser(modelJsonPath, option);

                std::cout << modelName << "time consuming: " << RunXGBoostTest(parser, testCsvPath) << "\n";
            }
        }

        void RunXGBoostOptimizeTests()
        {
        }
 
        void RunXGBoostCorrectnessTests()
        {
            std::string testModuleNames[] = {"abalone"};

            for (auto modelName : testModuleNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto stateCsvPath = testModelsDir + "/" + modelName + ".prob.csv";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";
                auto answerCsvPath = testModelsDir + "/" + modelName + ".answer.csv";

                BuildOptions option;
                option.enable_flint = true;
                option.enable_ra = true;
                XGBoostParser parser(modelJsonPath, option, stateCsvPath);

                verifyXGBoostResult(parser, testCsvPath, answerCsvPath);
            }
        }
    }
}