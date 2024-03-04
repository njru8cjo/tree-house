#ifndef CSVREADER_H
#define CSVREADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>

namespace mlir
{
namespace utils 
{

class CSVReader {
    std::vector<std::vector<double>> m_data;
    std::string m_filename;
public:
    CSVReader(const std::string& filename);
    std::vector<double>& getRow(size_t index) { return m_data[index]; }
    
    template<typename T>
    std::vector<T> getRowOfType(size_t index) {
        auto& doubleRow = getRow(index);
        std::vector<T> row(doubleRow.size());
        std::copy(std::begin(doubleRow), std::end(doubleRow), std::begin(row));
        return row;
    }

    size_t NumberOfRows() { return m_data.size(); }
};

std::vector<double> getNextLineAndSplitIntoTokens(std::istream& str) {
    std::vector<double> result;
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string cell;

    while(std::getline(lineStream, cell, ',')) {
        result.push_back(std::stof(cell));
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        result.push_back(NAN);
    }
    return result;
}

CSVReader::CSVReader(const std::string& filename) {
    std::ifstream fin(filename);
    assert(fin);
    while (!fin.eof()) {
        auto row = getNextLineAndSplitIntoTokens(fin);
        m_data.push_back(row);
    }
}

}
}

#endif