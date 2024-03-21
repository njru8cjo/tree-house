#ifndef SKLERAN_PARSER_H
#define SKLERAN_PARSER_H

#include "jsonparser.h"
#include <queue>

using json = nlohmann::json;

namespace Treehierarchy
{
    class SklearnParser : public JsonParser
    {
    public:
        SklearnParser(const std::string &forestJSONPath, BuildOptions option) : JsonParser(forestJSONPath, option) {}

        ~SklearnParser() {}

        void ConstructForest() override;
        void ConstructTree(const json treeJSON);

    private:
        void CreatePredictFunction() override;
        struct Node
        {
            json node;
            int64_t parent;
            double prob;
            bool isLeft;
        };
    };

    void SklearnParser::ConstructForest()
    {
        for (auto &treeJSON : m_json)
        {
            m_decisionTree = &(m_forest->newTree());
            ConstructTree(treeJSON);
        }
    }

    void SklearnParser::ConstructTree(const json treeJSON)
    {
        int64_t id;
        std::queue<Node> nodeQueue;
        Node currentNode;

        nodeQueue.push({treeJSON, DecisionTree::ROOT_NODE_PARENT, DecisionTree::ROOT_NODE_PROB, true});

        while (!nodeQueue.empty())
        {
            currentNode = nodeQueue.front();

            if (currentNode.node.value("isCategorical", "True") == "False")
            {
                int32_t featureIndex = currentNode.node["feature"].get<int32_t>();
                double threshold = currentNode.node["split"].get<double>();
                double probLeft = currentNode.node["probLeft"].get<double>();
                double probRight = currentNode.node["probRight"].get<double>();
                id = m_decisionTree->NewNode(threshold, featureIndex, currentNode.prob);

                nodeQueue.push({currentNode.node["leftChild"], id, probLeft * currentNode.prob, true});
                nodeQueue.push({currentNode.node["rightChild"], id, probRight * currentNode.prob, false});
            }
            else
            {
                std::vector<float> vec = currentNode.node["prediction"].get<std::vector<float>>();
                auto maxIt = std::max_element(vec.begin(), vec.end());
                double prediction = std::distance(vec.begin(), maxIt);
                m_forest->SetClassNum(vec.size()); // TODO: better way to get classNum?

                id = m_decisionTree->NewNode(prediction, DecisionTree::LEAF_NODE_FEATURE, currentNode.prob);
            }

            int64_t parentId = currentNode.parent;

            m_decisionTree->SetNodeParent(id, parentId);

            if (currentNode.isLeft && parentId != DecisionTree::ROOT_NODE_PARENT)
            {
                m_decisionTree->SetNodeLeftChild(parentId, id);
            }
            else if (parentId != DecisionTree::ROOT_NODE_PARENT)
            {
                m_decisionTree->SetNodeRightChild(parentId, id);
            }

            nodeQueue.pop();
        }
    }

    // TODO: this should be finish
    void SklearnParser::CreatePredictFunction()
    {
    }
}

#endif