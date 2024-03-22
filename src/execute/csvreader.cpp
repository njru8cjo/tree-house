#include "csvreader.h"

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
        CSVReader::CSVReader(const std::string &filename)
        {
            std::ifstream fin(filename);
            assert(fin);
            while (!fin.eof())
            {
                auto row = GetNextLineAndSplitIntoTokens(fin);
                m_data.push_back(row);
            }
        }

       
        std::vector<double> CSVReader::GetNextLineAndSplitIntoTokens(std::istream &str)
        {
            std::vector<double> result;
            std::string line;
            std::getline(str, line);

            std::stringstream lineStream(line);
            std::string cell;

            while (std::getline(lineStream, cell, ','))
            {
                result.push_back(std::stof(cell));
            }
            // This checks for a trailing comma with no data after it.
            if (!lineStream && cell.empty())
            {
                // If there was a trailing comma then add an empty element.
                result.push_back(NAN);
            }
            return result;
        }
    }
}