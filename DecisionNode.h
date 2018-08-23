#ifndef DECISION_NODE_H
#define DECISION_NODE_H

#include <unordered_set>
#include <vector>

enum class Position { LEFT, RIGHT };

class DecisionNode {
public:
  DecisionNode(const std::string &name);
  std::string NAME;
  const static int NUM_ACTIONS = 2;

  void init();
  void calcStrategy(const double weight);
  const std::vector<double> &getStrategy() const { return m_strategy; }
  void calcAvgBetProbabilities();
  void updateRegretSum(size_t actionIndex, double regretVal) {
    m_regretSum[actionIndex] += regretVal;
  }

  std::vector<double> getAvgBetProbabilities() const;
  std::vector<double> getAvgBetErrors() const;
  double getLastAvgBetProb() const;

  friend std::ostream &operator<<(std::ostream &os, const DecisionNode &node);

private:
  std::vector<double> m_strategy;
  std::vector<double> m_strategySum;
  std::vector<double> m_regretSum;
  std::vector<std::vector<double>> m_avgBetProbabilitiesList;
  std::vector<double> *m_avgBetProbPtr;
};

#endif // DECISION_NODE_H