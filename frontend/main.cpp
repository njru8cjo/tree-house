#include "sklearnparser.h"
#include "xgboostparser.h"

int main() {
    // SklearnParser parser("../DT_1.json");
    std::string modelJsonPath = "../DT_1.json";
    SklearnParser parser = SklearnParser(modelJsonPath);
    parser.constructForest();
    parser.print();

    modelJsonPath = "../abalone_xgb_model_save.json";
    XGBoostParser parser2 = XGBoostParser(modelJsonPath);
    parser2.constructForest();
    parser2.print();

    return 0;
}



