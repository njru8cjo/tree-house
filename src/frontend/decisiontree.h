
#ifndef DECISIONFOREST_H
#define DECISIONFOREST_H

#include <algorithm>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <cassert>
#include <utility>

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
        void SetClassId(size_t classId) { m_classId = classId; }

        Node GetNode(int64_t idx) { return m_nodes[idx]; }
        size_t GetClassId() { return m_classId; }
        const std::vector<Node> &GetNodes() { return m_nodes; }

    private:
        size_t m_classId = 0;
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
        void SetRegNum(size_t regNum) { m_regNum = regNum; }
        void SetFeatureProb(std::pair<unsigned, float> input) 
        {
            bool found = false;
            for (auto& pair : m_featureProb) 
            {
                if (pair.first == input.first) 
                {
                    pair.second += input.second;
                    found = true;
                    break;
                }
            }
            if (!found) 
                m_featureProb.push_back(input);
        }
        void DumpFeatureProb()
        {
            for (const auto& pair : m_featureProb)
                std::cout << "Feature: " << pair.first << ", Porb:" << pair.second << std::endl;
        }
        void SortFeatureProb()
        {
            std::sort(m_featureProb.begin(), m_featureProb.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second > b.second;
            });
        }

        size_t GetClassNum() const { return m_classNum; }
        size_t GetFeatureSize() { return m_featureSize; }
        double GetInitialValue() const { return m_initialValue; }
        size_t GetRegNum() { return m_regNum; }
        PredictionTransformation GetObjective() const { return m_objective; }
        std::vector<size_t> GetTopFeature()
        {
            std::vector<size_t> res;
            
            m_regNum = (m_featureProb.size() < m_regNum) ? m_featureProb.size() : m_regNum;
            
            for(size_t i = 0; i < m_regNum; i++)
            {
                res.push_back(m_featureProb[i].first);
            }
                
            return res;
        }
        int GetGlobalIdxFromFeature(size_t feature)
        {
            for(size_t i = 0; i < m_regNum; i++)
            {
                if(m_featureProb[i].first == feature)
                    return i;
            }
            // Search fail
            return -1;
        }
        size_t GetTreeSize() { return m_trees.size(); }
        DecisionTree *GetTree(size_t i) { return m_trees[i].get(); }

    private:
        std::vector<std::shared_ptr<DecisionTree>> m_trees;
        std::vector<std::pair<unsigned, float>> m_featureProb; // First is the index, second is the probability
        size_t m_classNum;
        size_t m_featureSize;
        double m_initialValue = 0;
        PredictionTransformation m_objective;
        size_t m_regNum = 16;
    };
}
#endif