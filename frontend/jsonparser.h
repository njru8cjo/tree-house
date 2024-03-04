#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include "json.hpp"
#include "decisiontree.h"
#include <fstream>

using json = nlohmann::json;

class JsonParser
{
protected:
    DecisionForest *m_forest;
    DecisionTree *m_decisionTree;
    json m_json;

public:
    JsonParser(const std::string &forestJSONPath) : m_forest(new DecisionForest())
    {
        std::ifstream fin(forestJSONPath);
        assert(fin);
        fin >> m_json;
    }

    // Provide a virtual destructor definition
    virtual ~JsonParser()
    {
        delete m_forest;
    }

    virtual void print();
    virtual void constructForest() = 0;

    DecisionForest *getDecisionForest() const
    {
        return m_forest;
    }
};

void JsonParser::print()
{
    m_forest->print();
}

#endif