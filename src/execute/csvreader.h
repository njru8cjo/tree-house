#ifndef CSVREADER_H
#define CSVREADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>

namespace Treehierarchy
{
    namespace utils
    {

        class CSVReader
        {
            std::vector<std::vector<double>> m_data;
            std::string m_filename;

        public:
            CSVReader(const std::string &filename);
            std::vector<double> &GetRow(size_t index) { return m_data[index]; }

            template <typename T>
            std::vector<T> GetRowOfType(size_t index)
            {
                auto &doubleRow = GetRow(index);
                std::vector<T> row(doubleRow.size());
                std::copy(std::begin(doubleRow), std::end(doubleRow), std::begin(row));
                return row;
            }

            size_t GetRowNum() { return m_data.size(); }
            std::vector<double> GetNextLineAndSplitIntoTokens(std::istream &str);
        };

    }
}

#endif