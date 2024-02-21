#include "jsonparser.h"
#include <queue>

using json = nlohmann::json;

class SklearnParser : public JsonParser {
public:
    SklearnParser(const std::string& treeJSON) : JsonParser(treeJSON) {}

    ~SklearnParser() {}

    //void print() override;
    void constructForest() override;
    void constructTree(const json treeJSON);

private:
    struct Node {
        json node;
        int64_t parent;
        double prob;
        bool isLeft;
    };
};

void SklearnParser::constructForest()
{
    for (auto& treeJSON : m_json)
    {
        m_decisionTree = &(m_forest->newTree());
        constructTree(treeJSON);
    }
}

void SklearnParser::constructTree(const json treeJSON)
{
    int64_t id;
    std::queue<Node> nodeQueue;
    Node currentNode;

    nodeQueue.push({treeJSON, DecisionTree::ROOT_NODE_PARENT, DecisionTree::ROOT_NODE_PROB, true});

    while (!nodeQueue.empty())
    {
        currentNode = nodeQueue.front();

        if(currentNode.node.value("isCategorical", "True") == "False") {
            int32_t featureIndex = currentNode.node["feature"].get<int32_t>();
            double threshold = currentNode.node["split"].get<double>();
            double probLeft = treeJSON["probLeft"].get<double>();
            double probRight = treeJSON["probRight"].get<double>();

            id = m_decisionTree->NewNode(threshold, featureIndex, currentNode.prob);

            nodeQueue.push({currentNode.node["leftChild"], id, probLeft * currentNode.prob, true});
            nodeQueue.push({currentNode.node["rightChild"], id, probRight * currentNode.prob, false});
        } else {
            std::vector<float> vec = currentNode.node["prediction"].get<std::vector<float>>();
            auto maxIt = std::max_element(vec.begin(), vec.end());
            double prediction = std::distance(vec.begin(), maxIt);
            
            id = m_decisionTree->NewNode(prediction, DecisionTree::LEAF_NODE_FEATURE, currentNode.prob);            
        }

        int64_t parentId = currentNode.parent;

        m_decisionTree->SetNodeParent(id, parentId);

        if(currentNode.isLeft && parentId != DecisionTree::ROOT_NODE_PARENT) {
            m_decisionTree->SetNodeLeftChild(parentId, id);
        } else if(parentId != DecisionTree::ROOT_NODE_PARENT) {
            m_decisionTree->SetNodeRightChild(parentId, id);
        }
       

        nodeQueue.pop();
    }
}

