#ifndef GAME_H
#define GAME_H

#include "DecisionNode.h"

#include <chrono>
#include <map>
#include <omp.h>
#include <random>
#include <set>
#include <vector>

enum class Card { J = 0, Q = 1, K = 2, A = 3 };
enum class Action { P = 0, B = 1 };

class Game {

public:
  Game(const size_t nPlayers);

  void train(size_t iterations);
  void cfrm();
  void init();
  void calcProperties(size_t numSamples);
  auto &getTotalAvgNodesReached() const { return m_totalAvgNodesReached; }
  auto &getAvgNodesReached() const { return m_avgNodesReached; }
  auto &getAvgCalcTimes() const { return m_avgCalcTimes; }
  auto &getAvgBetProbabilities() const { return m_avgBetProbabilities; }
  auto &getAvgBetErrors() const { return m_avgBetErrors; }
  auto &getAvgGameValues() const { return m_avgGameValues; }
  auto &getAvgGameValueErrors() const { return m_avgGameValueErrors; }
  auto &getExpectedGameValues() const { return m_expectedGameValues; }
  auto &getGameValueDists() const { return m_gameValueDists; }
  auto &getNodesReachedList() const { return m_nodesReachedList; }
  std::vector<std::string> getDNodeNames() const;

  friend std::ostream &operator<<(std::ostream &os, const Game &game);

private:
  std::vector<std::vector<Card>> findCardPermutations(size_t k) const;
  void findCardCombinations(std::vector<std::vector<Card>> &permutations,
                            std::vector<Card> &currentPermutation, size_t k,
                            size_t index, size_t start, size_t end) const;
  size_t showdown(const std::vector<Card> &cards,
                  const std::vector<size_t> &playerIndexes) const;

  Action selectRandomAction(const std::vector<double> &strategy);
  void selectRandomCards();
  void calcPlayerInvestments(std::vector<double> &investments,
                             const std::string &history) const;
  void calcTerminalUtilities(std::vector<double> &utilities,
                             const std::vector<Card> &cards,
                             const std::string &history) const;
  size_t calcCardSetIndex(const std::vector<Card> &cards) const;

  const size_t NUM_PLAYERS;
  const size_t NUM_CARDS;
  const size_t NUM_LEVELS;
  std::mt19937 m_rng;
  std::vector<Card> m_gameCards;
  std::vector<std::vector<std::vector<double>>> m_gameValues;
  std::vector<std::vector<double>> *m_gameValuesPtr;

  std::vector<std::vector<std::vector<DecisionNode>>> m_dNodes;
  std::vector<std::vector<std::vector<double>>> m_utilities;
  std::vector<std::vector<std::vector<std::vector<double>>>>
      m_terminalUtilities;
  std::vector<std::vector<size_t>> m_dNodeIndexes;
  std::vector<std::vector<bool>> m_isDNode;
  std::vector<std::vector<std::vector<double>>> m_weights;
  std::vector<std::vector<bool>> m_blacklist;

  std::vector<std::vector<double>> m_avgBetProbabilities;
  std::vector<std::vector<double>> m_avgBetErrors;
  std::vector<std::vector<double>> m_avgGameValues;
  std::vector<std::vector<double>> m_avgGameValueErrors;
  std::vector<double> m_expectedGameValues;
  std::vector<double> m_gameValueDists;
  std::vector<std::vector<double>> m_nodesReachedList;
  std::vector<double> m_avgNodesReached;
  double m_currentNodesReached;
  double m_totalAvgNodesReached;
  std::vector<std::vector<double>> m_calcTimesList;
  std::vector<double> m_avgCalcTimes;
  std::chrono::time_point<std::chrono::steady_clock> m_startTime;
};

#endif // GAME_H