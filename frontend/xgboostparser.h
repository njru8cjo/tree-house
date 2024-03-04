#include "jsonparser.h"
#include "csvreader.h"

using json = nlohmann::json;

class XGBoostParser : public JsonParser
{
public:
    XGBoostParser(const std::string &forestJSONPath) : JsonParser(forestJSONPath) {}
    XGBoostParser(const std::string &forestJSONPath, const std::string &statFilePath) : JsonParser(forestJSONPath), m_statFilePath(statFilePath) {}
    ~XGBoostParser() {}

    void constructForest() override;
    void constructTree(const json treeJSON);

private:
    void readProbabilityProfile();
    std::string m_statFilePath;
};

void XGBoostParser::constructForest()
{
    auto &learnerJSON = m_json["learner"];

    auto &featureTypesJSON = learnerJSON["feature_types"];
    m_forest->setFeatureSize(featureTypesJSON.size());

    auto &boosterJSON = learnerJSON["gradient_booster"];
    auto &modelJSON = boosterJSON["model"];
    auto &treesJSON = modelJSON["trees"];

    for (auto &treeJSON : treesJSON)
    {
        m_decisionTree = &(m_forest->newTree());
        constructTree(treeJSON);
    }

    if (m_statFilePath != "")
        readProbabilityProfile();
}

void XGBoostParser::readProbabilityProfile()
{
    utils::CSVReader reader(m_statFilePath);

    for (size_t i = 0; i < m_forest->getTreeSize(); i++)
    {

        auto row = reader.getRow(i + 1);
        auto tree = m_forest->getTree(i);
        auto nodes = tree->getNodes();

        std::vector<size_t> leafIndices;
        for (size_t j = 0; j < nodes.size(); j++)
        {
            if (nodes.at(j).isLeaf())
                leafIndices.push_back(j);
        }

        std::vector<int32_t> hitCounts(nodes.size(), 0);
        ;
        for (size_t j = 0; j < leafIndices.size(); j++)
        {
            int32_t hitCount = (int32_t)row.at(j * 2);

            auto node = tree->getNode(leafIndices.at(j));
            hitCounts[node.id] = hitCount;
            while (node.id != 0)
            {
                node = tree->getNode(node.parent);
                hitCounts[node.id] += hitCount;
            }
        }

        for (size_t i = 0; i < hitCounts.size(); i++)
        {
            double prob = (double)hitCounts[i] / hitCounts[0];
            tree->setProbability(i, prob);
        }
    }
}

void XGBoostParser::constructTree(const json treeJSON)
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
        int64_t nodeId = m_decisionTree->newNode(thresholds[i].get<double>(), featureIndices[i].get<int64_t>(), 1.0);
        nodeIds.push_back(nodeId);
    }

    for (size_t i = 0; i < num_nodes; i++)
    {
        auto leftChildIndex = left_children[i].get<int>();
        if (leftChildIndex != -1)
            m_decisionTree->setNodeLeftChild(nodeIds[i], nodeIds[leftChildIndex]);
        auto rightChildIndex = right_childen[i].get<int>();
        if (rightChildIndex != -1)
            m_decisionTree->setNodeRightChild(nodeIds[i], nodeIds[rightChildIndex]);

        auto parentIndex = parents[i].get<int>();
        if (parents[i].get<int>() == 2147483647)
            m_decisionTree->setNodeParent(nodeIds[i], DecisionTree::ROOT_NODE_PARENT);
        else
            m_decisionTree->setNodeParent(nodeIds[i], nodeIds[parentIndex]);
    }
}