#ifndef DECISION_NODE_H
#define DECISION_NODE_H

#include "DataTypes.h"

#include <unordered_set>
#include <vector>

class DecisionNode {
public:
  DecisionNode(const std::string &name);

  const Vec1D<double>& getStrategy() const { return m_strategy; }
  std::string getName() const { return m_name; }

  void init();
  void calcStrategy(const double weight);
  void calcAvgBetProbabilities();
  void updateRegretSum(size_t actionIndex, double regretVal) {
    m_regretSum[actionIndex] += regretVal;
  }

  Vec1D<double> getAvgBetProbabilities() const;
  Vec1D<double> getAvgBetErrors() const;
  double getLastAvgBetProb() const;

  friend std::ostream &operator<<(std::ostream &os, const DecisionNode &node);

private:
  Vec1D<double> m_strategy;
  Vec1D<double> m_strategySum;
  Vec1D<double> m_regretSum;
  Vec2D<double> m_avgBetProbabilitiesList;
  Vec1D<double> *m_avgBetProbPtr;
  std::string m_name;
};

#endif // DECISION_NODE_H