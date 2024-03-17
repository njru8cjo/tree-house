
#ifndef DECISIONFOREST_H
#define DECISIONFOREST_H

#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <cassert>

namespace Treehierarchy
{
    enum class PredictionTransformation
    {
        kIdentity,
        kSigmoid,
        kSoftMax
    };

    class DecisionTree
    {
    public:
        static constexpr int64_t EMPTY_NODE_INDEX = -1;
        static constexpr int64_t LEAF_NODE_FEATURE = -1;
        static constexpr int64_t ROOT_NODE_PARENT = -1;
        static constexpr double ROOT_NODE_PROB = 1.0;

        DecisionTree()
        {
        }

        struct Node
        {
            int64_t id;
            double threshold;
            int32_t featureIndex;
            int64_t parent = ROOT_NODE_PARENT;
            int64_t leftChild;
            int64_t rightChild;
            double probability;

            bool operator==(const Node &that) const
            {
                return id == that.id;
            }

            bool IsLeaf() const
            {
                return leftChild == EMPTY_NODE_INDEX && rightChild == EMPTY_NODE_INDEX;
            }
        };

        int64_t NewNode(double threshold, int32_t featureIndex, double probability)
        {
            Node node{(int64_t)m_nodes.size(), threshold, featureIndex, ROOT_NODE_PARENT, EMPTY_NODE_INDEX, EMPTY_NODE_INDEX, probability};
            m_nodes.push_back(node);

            return node.id;
        }

        void SetNodeParent(int64_t idx, int64_t parent) { m_nodes[idx].parent = parent; }
        void SetNodeRightChild(int64_t idx, int64_t child) { m_nodes[idx].rightChild = child; }
        void SetNodeLeftChild(int64_t idx, int64_t child) { m_nodes[idx].leftChild = child; }
        void SetProbability(int64_t idx, double prob) { m_nodes[idx].probability = prob; }

        Node GetNode(int64_t idx) { return m_nodes[idx]; }
        const std::vector<Node> &GetNodes() { return m_nodes; }

    private:
        std::vector<Node> m_nodes;
    };

    class DecisionForest
    {
    public:
        DecisionForest() {}

        DecisionTree &newTree()
        {
            m_trees.push_back(std::make_shared<DecisionTree>());
            return *(m_trees.back());
        }

        void SetClassNum(size_t classNum) { m_classNum = classNum; }
        void SetFeatureSize(size_t size) { m_featureSize = size; }
        void SetInitialValue(double value) { m_initialValue = value; }
        void SetObjective(PredictionTransformation val) { m_objective = val; }

        size_t GetClassNum() const { return m_classNum; }
        size_t GetFeatureSize() { return m_featureSize; }
        double GetInitialValue() const { return m_initialValue; }
        PredictionTransformation GetObjective() const { return m_objective; }

        size_t GetTreeSize() { return m_trees.size(); }
        DecisionTree *GetTree(size_t i) { return m_trees[i].get(); }

    private:
        std::vector<std::shared_ptr<DecisionTree>> m_trees;
        size_t m_classNum;
        size_t m_featureSize;
        double m_initialValue = 0;
        PredictionTransformation m_objective;
    };
}
#endif