#include "DecisionNode.h"

#include <algorithm>
#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

constexpr size_t NUM_ACTIONS = 2;

/* Constructor
 */
DecisionNode::DecisionNode(const std::string &name)
    : m_name(name),
      m_strategy(NUM_ACTIONS), 
      m_strategySum(NUM_ACTIONS),
      m_regretSum(NUM_ACTIONS),
      m_avgBetProbPtr(NULL) {}

/* Initialise the decision node's state and reset all strategies and regrets
 */
void DecisionNode::init() {
  std::fill(m_strategy.begin(), m_strategy.end(), 0.0);
  std::fill(m_strategySum.begin(), m_strategySum.end(), 0.0);
  std::fill(m_regretSum.begin(), m_regretSum.end(), 0.0);

  m_avgBetProbabilitiesList.push_back(Vec1D<double>());
  m_avgBetProbPtr = &m_avgBetProbabilitiesList.back();
}

/* Calculate the strategy to use at this decision node based on past regrets
 * @return :: The calculated strategy
 */
void DecisionNode::calcStrategy(const double weight) {
  // Set strategy equal to all positive regrets
  Vec1D<double> positiveRegrets(NUM_ACTIONS);
  for (size_t i = 0; i < NUM_ACTIONS; ++i)
    positiveRegrets[i] = std::max(m_regretSum[i], 0.0);

  double normalizingSum =
      std::accumulate(positiveRegrets.begin(), positiveRegrets.end(), 0.0);
  for (size_t i = 0; i < NUM_ACTIONS; i++) {
    // Normalize all positive regrets if we have a non-zero normalizing sum,
    // otherwise do not alter strategy
    m_strategy[i] = (normalizingSum > 0) ? positiveRegrets[i] / normalizingSum
                                         : 1.0 / NUM_ACTIONS;
    m_strategySum[i] += m_strategy[i] * weight;
  }
}

/* Calculate the average bet probabilities for this node
 */
void DecisionNode::calcAvgBetProbabilities() {
  Vec1D<double> avgStrategy(NUM_ACTIONS);
  double normalizingSum =
      std::accumulate(m_strategySum.begin(), m_strategySum.end(), 0.0);
  for (size_t i = 0; i < NUM_ACTIONS; ++i) {
    // Normalize all strategy sums if we have a non-zero normalizing sum,
    // otherwise assign a uniform strategy
    avgStrategy[i] = (normalizingSum > 0) ? m_strategySum[i] / normalizingSum
                                          : 1.0 / NUM_ACTIONS;
  }
  m_avgBetProbPtr->push_back(avgStrategy[1]);
}

/* Obtains the averaged set of bet probabilities from the average bet
 * probabilities list
 * @return :: Vector of averaged bet probabilities
 */
Vec1D<double> DecisionNode::getAvgBetProbabilities() const {
  const size_t numSets = m_avgBetProbabilitiesList.size();
  const size_t numProbabilities = m_avgBetProbabilitiesList.front().size();
  Vec1D<double> avgBetProbabilities(numProbabilities);

  for (size_t i = 0; i < numProbabilities; ++i) {
    double normalizingSum = 0.0;
    for (const auto &probSet : m_avgBetProbabilitiesList)
      normalizingSum += probSet[i];

    avgBetProbabilities[i] = normalizingSum / numSets;
  }

  return avgBetProbabilities;
}

/* Obtains the errors for the averaged sets of bet probabilities
 * @return :: Vector of averaged bet probability errors
 */
Vec1D<double> DecisionNode::getAvgBetErrors() const {
  const size_t numSets = m_avgBetProbabilitiesList.size();
  const size_t numProbabilities = m_avgBetProbabilitiesList.front().size();
  const auto &avgBetProbabilities = getAvgBetProbabilities();
  Vec1D<double> avgBetErrors(numProbabilities);

  for (size_t i = 0; i < numProbabilities; ++i) {
    double squareSum = 0.0;
    for (size_t j = 0; j < numSets; ++j)
      squareSum +=
          pow(m_avgBetProbabilitiesList[j][i] - avgBetProbabilities[i], 2.0);

    avgBetErrors[i] = sqrt(squareSum / numSets);
  }

  return avgBetErrors;
}

/* Obtains the last stored average bet probability. If no previous bet
 * probabilities exist, a uniform betting probability is returned.
 * @return :: Last average bet probability value
 */
double DecisionNode::getLastAvgBetProb() const {
  return m_avgBetProbPtr->empty() ? 1.0 / NUM_ACTIONS : m_avgBetProbPtr->back();
}

/* Output this node's name and last average strategy
 */
std::ostream &operator<<(std::ostream &os, const DecisionNode &node) {
  using boost::lexical_cast;
  using boost::algorithm::join;

  // Display node name with pass and bet probabilities
  const auto &avgBetProbabilities = node.getAvgBetProbabilities();
  if (avgBetProbabilities.empty())
    return os << "[0.000000, 0.000000]";

  Vec1D<std::string> strVec;
  std::stringstream ss1, ss2;
  ss1 << std::fixed << std::setprecision(6) << 1.0 - avgBetProbabilities.back();
  ss2 << std::fixed << std::setprecision(6) << avgBetProbabilities.back();
  strVec.push_back(ss1.str());
  strVec.push_back(ss2.str());

  return os << "[" << join(strVec, ",") << "]";
}