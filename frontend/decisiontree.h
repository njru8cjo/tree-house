
#ifndef DECISIONFOREST_H
#define DECISIONFOREST_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

class DecisionTree
{
public:
    static constexpr int64_t EMPTY_NODE_INDEX   = -1;
    static constexpr int64_t LEAF_NODE_FEATURE  = -1;
    static constexpr int64_t ROOT_NODE_PARENT   = -1;
    static constexpr double ROOT_NODE_PROB      = 1.0;

    DecisionTree() {
        
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
        
        bool operator==(const Node& that) const
        {
            return id==that.id;
        }

        bool isLeaf() const
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

    void SetNodeParent(int64_t node, int64_t parent) { m_nodes[node].parent = parent; }
    void SetNodeRightChild(int64_t node, int64_t child) { m_nodes[node].rightChild = child; }
    void SetNodeLeftChild(int64_t node, int64_t child) { m_nodes[node].leftChild = child; }

    void print();

private:
    std::vector<Node> m_nodes;
};

class DecisionForest
{
public:
    DecisionForest() {}
    
    DecisionTree& newTree() { 
        m_trees.push_back(std::make_shared<DecisionTree>());
        return *(m_trees.back());
    }
    
    void print();

private:
    std::vector<std::shared_ptr<DecisionTree>> m_trees;
};


void DecisionTree::print() {

    std::cout << "nodes number =" << m_nodes.size() << "\n";

    for (Node node : m_nodes) {
        std::cout << "node " << node.id << " parent:" << node.parent << " leftchild:" << node.leftChild  << " rightchild:" << node.rightChild 
        << " split=" << node.threshold << " feature=" << node.featureIndex << " probability=" << node.probability << " " << std::endl;
    }
    std::cout << "===================\n";    
}


void DecisionForest::print() {        
    for(auto& tree: m_trees) {
        tree->print();
    }
}

#endif