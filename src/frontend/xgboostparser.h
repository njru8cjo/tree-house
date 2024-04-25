#ifndef XGBOOST_PARSER_H
#define XGBOOST_PARSER_H

#include "jsonparser.h"
#include "csvreader.h"
#include <utility>

using json = nlohmann::json;

namespace Treehierarchy
{
    class XGBoostParser : public JsonParser
    {
    public:
        XGBoostParser(const std::string &forestJSONPath, BuildOptions option) : JsonParser(forestJSONPath, option) {}
        XGBoostParser(const std::string &forestJSONPath, BuildOptions option, const std::string &statFilePath) : JsonParser(forestJSONPath, option), m_statFilePath(statFilePath) {}
        ~XGBoostParser() {}

        void ConstructForest() override;
        void ConstructTree(const json treeJSON);

    protected:
        void ReadProbabilityProfile();
        double TransformBaseScore(const PredictionTransformation objective, double val);
        PredictionTransformation GetPredictionTransformType(const std::string &objectiveName);
        void CreatePredictFunction() override;
        void CreateLeafNode(Value result, DecisionTree::Node node) override;

        arith::CmpFPredicate getComparePredicate() override { return arith::CmpFPredicate::OLT; }
        arith::CmpFPredicate getReverseComparePredicate() override { return arith::CmpFPredicate::OGE; }
        LLVM::ICmpPredicate getCompareIntPredicate() override { return LLVM::ICmpPredicate::slt; }
        LLVM::ICmpPredicate getReverseCompareIntPredicate() override { return LLVM::ICmpPredicate::sge; }
    
        FunctionType getTreeFunctionType() override {             
            Type argType = getFeaturePointerType();
            Type resultType = getResultPointerType();
            return m_builder.getFunctionType({argType, resultType}, getF32());
        };


        std::string m_statFilePath;
    };

    void XGBoostParser::ConstructForest()
    {
        auto &learnerJSON = m_json["learner"];

        auto &featureTypesJSON = learnerJSON["feature_types"];
        m_forest->SetFeatureSize(featureTypesJSON.size());

        auto objective = GetPredictionTransformType(learnerJSON["objective"]["name"].get<std::string>());
        m_forest->SetObjective(objective);

        auto classNum = std::stoi(learnerJSON["learner_model_param"]["num_class"].get<std::string>());
        classNum = classNum == 0 ? 1 : classNum;
        m_forest->SetClassNum(classNum);

        auto baseScore = std::stod(learnerJSON["learner_model_param"]["base_score"].get<std::string>());
        m_forest->SetInitialValue(TransformBaseScore(objective, baseScore));

        auto &boosterJSON = learnerJSON["gradient_booster"];
        auto &modelJSON = boosterJSON["model"];
        auto &treeInfoJSON = modelJSON["tree_info"];
        auto &treesJSON = modelJSON["trees"];

        int32_t treeIndex = 0;
        for (auto &treeJSON : treesJSON)
        {
            m_decisionTree = &(m_forest->newTree());
            ConstructTree(treeJSON);
            m_decisionTree->SetClassId(treeInfoJSON[treeIndex++]);
        }

        if (m_statFilePath != "")
            ReadProbabilityProfile();

        m_forest->SortFeatureProb();
    }

    void XGBoostParser::ConstructTree(const json treeJSON)
    {

        auto &left_children = treeJSON["left_children"];
        auto &right_childen = treeJSON["right_children"];
        auto &parents = treeJSON["parents"];
        auto &thresholds = treeJSON["split_conditions"];
        auto &featureIndices = treeJSON["split_indices"];

        auto num_nodes = static_cast<size_t>(std::stoi(treeJSON["tree_param"]["num_nodes"].get<std::string>()));
        std::vector<int64_t> nodeIds;

        for (size_t i = 0; i < num_nodes; i++)
        {
            int64_t nodeId = m_decisionTree->NewNode(thresholds[i].get<double>(), featureIndices[i].get<int64_t>(), 1.0);
            nodeIds.push_back(nodeId);
        }

        for (size_t i = 0; i < num_nodes; i++)
        {
            auto leftChildIndex = left_children[i].get<int>();
            if (leftChildIndex != -1)
                m_decisionTree->SetNodeLeftChild(nodeIds[i], nodeIds[leftChildIndex]);
            auto rightChildIndex = right_childen[i].get<int>();
            if (rightChildIndex != -1)
                m_decisionTree->SetNodeRightChild(nodeIds[i], nodeIds[rightChildIndex]);

            auto parentIndex = parents[i].get<int>();
            if (parents[i].get<int>() == 2147483647)
                m_decisionTree->SetNodeParent(nodeIds[i], DecisionTree::ROOT_NODE_PARENT);
            else
                m_decisionTree->SetNodeParent(nodeIds[i], nodeIds[parentIndex]);
        }
    }

    void XGBoostParser::ReadProbabilityProfile()
    {
        utils::CSVReader reader(m_statFilePath);

        for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
        {

            auto row = reader.GetRow(i + 1);
            auto tree = m_forest->GetTree(i);
            auto nodes = tree->GetNodes();

            std::vector<size_t> leafIndices;
            for (size_t j = 0; j < nodes.size(); j++)
            {
                if (nodes.at(j).IsLeaf())
                    leafIndices.push_back(j);
            }

            std::vector<int32_t> hitCounts(nodes.size(), 0);
            ;
            for (size_t j = 0; j < leafIndices.size(); j++)
            {
                int32_t hitCount = (int32_t)row.at(j * 2);

                auto node = tree->GetNode(leafIndices.at(j));
                hitCounts[node.id] = hitCount;
                while (node.id != 0)
                {
                    node = tree->GetNode(node.parent);
                    hitCounts[node.id] += hitCount;
                }
            }

            for (size_t i = 0; i < hitCounts.size(); i++)
            {
                double prob = (double)hitCounts[i] / hitCounts[0];
                tree->SetProbability(i, prob);
                m_forest->SetFeatureProb(std::make_pair(tree->GetNode(i).featureIndex, prob));
            }
        }
    }

    PredictionTransformation XGBoostParser::GetPredictionTransformType(const std::string &objectiveName)
    {
        if (objectiveName == "binary:logistic")
            return PredictionTransformation::kSigmoid;
        else if (objectiveName == "reg:squarederror")
            return PredictionTransformation::kIdentity;
        else if (objectiveName == "multi:softmax")
            return PredictionTransformation::kSoftMax;
        else
            assert(false && "Unknown objective type");
    }

    double XGBoostParser::TransformBaseScore(const PredictionTransformation objective, double val)
    {
        if (objective == PredictionTransformation::kSigmoid)
            return -log(1.0 / val - 1.0);
        else if (objective == PredictionTransformation::kIdentity || objective == PredictionTransformation::kSoftMax)
            return val;
        else
            assert(false && "Unknown objective type");
        return val;
    }

    void XGBoostParser::CreatePredictFunction()
    {
        Location loc = m_builder.getUnknownLoc();

        Type argType = getFeaturePointerType();
        auto functionType = m_builder.getFunctionType({argType, argType}, getF32());
        auto mainFun = m_builder.create<func::FuncOp>(loc, "predict", functionType);
        mainFun.setPublic();

        Block *callerBlock = mainFun.addEntryBlock();
        m_builder.setInsertionPointToStart(callerBlock);

        Value input = callerBlock->getArgument(1);
        Value result[m_forest->GetClassNum()];

        // If RA, load pin data to global variable
        if(m_option.enable_ra) 
        {
            // Get features
            std::vector<size_t> pin_features = m_forest->GetTopFeature();
            size_t regNum = m_forest->GetRegNum();
            for(size_t i = 0; i < regNum; i++)
            {
                // Load data
                Value featureIdx = m_builder.create<arith::ConstantIntOp>(loc, pin_features[i], getI32());
                Value featurePtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getFeatureType(), callerBlock->getArgument(0), featureIdx);
                pin_reg[i] = m_builder.create<LLVM::LoadOp>(loc, getFeatureType(), featurePtr);
            }  
            Block *curBlock = m_builder.getBlock();
            for (size_t i = 0; i < m_forest->GetClassNum(); i++)
            {
                auto zeroConst = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(m_forest->GetInitialValue()));
                Value idx = m_builder.create<arith::ConstantIntOp>(loc, i, getI32());
                Value resultPtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getF32(), input, idx);
                m_builder.create<LLVM::StoreOp>(loc, zeroConst, resultPtr);
            }
            for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
            {
                // Get tree
                DecisionTree *tree = m_forest->GetTree(i);
                // Get result class
                auto classId = m_forest->GetTree(i)->GetClassId();
                Value idx = m_builder.create<arith::ConstantIntOp>(loc, classId, getI32());
                Value resultPtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getF32(), input, idx);
                buildRANodeOp(curBlock, tree, 0, resultPtr);
                // Coacate next root to previous leaf
                Block *nextBlock = m_builder.createBlock(&mainFun.getBody());
                for(Block* lb: leafBlock)
                {
                    m_builder.setInsertionPointToEnd(lb);
                    m_builder.create<LLVM::BrOp>(loc, nextBlock);
                }
                leafBlock.clear();
                m_builder.setInsertionPointToStart(nextBlock);
            }
            for (size_t i = 0; i < m_forest->GetClassNum(); i++)
            {
                for (size_t i = 0; i < m_forest->GetClassNum(); i++)
                {
                    result[i] = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(m_forest->GetInitialValue()));
                }
                if (m_forest->GetObjective() == PredictionTransformation::kSigmoid)
                {
                    Value idx = m_builder.create<arith::ConstantIntOp>(loc, i, getI32());
                    Value resultPtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getF32(), input, idx);
                    Value loadData = m_builder.create<LLVM::LoadOp>(loc, getF32(), resultPtr);
                    auto negate = m_builder.create<arith::NegFOp>(loc, getF32(), loadData);
                    auto exponential = m_builder.create<math::ExpOp>(loc, getF32(), static_cast<Value>(negate));
                    auto oneConst = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr((float)1.0));
                    auto onePlusExp = m_builder.create<arith::AddFOp>(loc, getF32(), oneConst, exponential);
                    auto divRes = m_builder.create<arith::DivFOp>(loc, getF32(), oneConst, onePlusExp);
                    m_builder.create<LLVM::StoreOp>(loc, divRes, resultPtr);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
            {            
                SmallVector<Value> operands = {callerBlock->getArgument(0), callerBlock->getArgument(1)};
                auto callResult = m_builder.create<func::CallOp>(loc, StringRef("tree_" + std::to_string(i)), getF32(), operands);
                auto classId = m_forest->GetTree(i)->GetClassId();
                result[classId] = m_builder.create<arith::AddFOp>(loc, result[classId], callResult.getResult(0));
            }
            for (size_t i = 0; i < m_forest->GetClassNum(); i++)
            {

                if (m_forest->GetObjective() == PredictionTransformation::kSigmoid)
                {
                    auto negate = m_builder.create<arith::NegFOp>(loc, getF32(), result[i]);
                    auto exponential = m_builder.create<math::ExpOp>(loc, getF32(), static_cast<Value>(negate));
                    auto oneConst = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr((float)1.0));
                    auto onePlusExp = m_builder.create<arith::AddFOp>(loc, getF32(), oneConst, exponential);
                    result[i] = m_builder.create<arith::DivFOp>(loc, getF32(), oneConst, onePlusExp);
                }

                Value idx = m_builder.create<arith::ConstantIntOp>(loc, i, getI32());
                Value resultPtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getF32(), input, idx);
                m_builder.create<LLVM::StoreOp>(loc, result[i], resultPtr);
            }
        }

        
        
        m_builder.create<func::ReturnOp>(loc, result[0]);
        m_module.push_back(mainFun);
    }

    void XGBoostParser::CreateLeafNode(Value result, DecisionTree::Node node) 
    {
        auto loc = m_builder.getUnknownLoc();
        Value retVal = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(node.threshold));
        m_builder.create<func::ReturnOp>(loc, retVal);
    }

}

#endif