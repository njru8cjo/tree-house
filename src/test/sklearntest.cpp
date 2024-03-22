#include <libgen.h>
#include <unistd.h>

#include "csvreader.h"
#include "modulerunner.h"
#include "sklearnparser.h"

namespace Treehierarchy
{
    namespace test
    {
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

        void RunSKlearnNonOptimizeTests()
        {
            std::puts("Running Without Optimize...\n");
            auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
            auto modelJsonPath = testModelsDir + "/DT_1.json";
            auto testCsvPath = testModelsDir + "/DT_1.test.csv";

            BuildOptions option;
            SklearnParser parser(modelJsonPath, option);

            parser.ConstructForest();

            ModuleOp module = parser.buildHIRModule();
            module = parser.lowerToLLVMModule();

            module->dump();

            Execute::ModuleRunner runner(module);

            utils::CSVReader testCaseReader(testCsvPath);
            std::vector<std::vector<float>> inputData;

            for (size_t i = 0; i < testCaseReader.GetRowNum(); i++)
            {
                auto row = testCaseReader.GetRowOfType<float>(i);
                inputData.push_back(row);
            }

            // utils::CSVReader answerReader(answerCsvPath);
            // auto epsilon = answerReader.GetRowOfType<float>(0)[0];
            // auto answers = answerReader.GetRowOfType<float>(1);

           //int i = 0;
            auto classNum = parser.getForestClassNum();
            std::vector<float> result(classNum, 0);

            for (auto &input : inputData) 
            {
                runner.runInference(input.data(), result.data());
            }
            for(size_t i = 0; i < classNum; i++) 
                std::cout << result[i] << "\t";
            std::cout << "\n";
        }
    }
}