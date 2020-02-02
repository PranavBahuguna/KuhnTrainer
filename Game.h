#ifndef GAME_H
#define GAME_H

#include "DataTypes.h"
#include "DecisionNode.h"

#include <chrono>
#include <map>
#include <omp.h>
#include <random>
#include <set>
#include <vector>

enum class Card { J = 0, Q = 1, K = 2, A = 3 };
enum class Action { P = 0, B = 1 };

typedef DecisionNode DNode;

class Game {

public:
  Game(const int nPlayers);

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
  Vec1D<std::string> getDNodeNames() const;

  friend std::ostream &operator<<(std::ostream &os, const Game &game);

private:
  Vec2D<Card> findCardPermutations(size_t k) const;
  void findCardCombinations(Vec2D<Card> &permutations, Vec1D<Card> &currentPermutation, size_t k, size_t index, size_t start, size_t end) const;
  size_t showdown(const Vec1D<Card> &cards, const Vec1D<size_t> &playerIndexes) const;
  Action selectRandomAction(const Vec1D<double> &strategy);
  void selectRandomCards();
  void calcPlayerInvestments(Vec1D<double> &investments,const std::string &history) const;
  void calcTerminalUtilities(Vec1D<double> &utilities, const Vec1D<Card> &cards,const std::string &history) const;
  size_t calcCardSetIndex(const Vec1D<Card> &cards) const;

  const size_t NUM_PLAYERS;
  const size_t NUM_CARDS;
  const size_t NUM_LEVELS;

  std::mt19937 m_rng;
  Vec1D<Card> m_gameCards;
  Vec3D<double> m_gameValues;
  Vec2D<double> *m_gameValuesPtr;

  Vec3D<DNode> m_dNodes;
  Vec3D<double> m_utilities;
  Vec4D<double> m_terminalUtilities;
  Vec2D<int> m_dNodeIndexes;
  Vec2D<bool> m_isDNode;
  Vec3D<double> m_weights;
  Vec2D<bool> m_blacklist;

  Vec2D<double> m_avgBetProbabilities;
  Vec2D<double> m_avgBetErrors;
  Vec2D<double> m_avgGameValues;
  Vec2D<double> m_avgGameValueErrors;
  Vec1D<double> m_expectedGameValues;
  Vec1D<double> m_gameValueDists;
  Vec2D<double> m_nodesReachedList;
  Vec1D<double> m_avgNodesReached;
  double m_currentNodesReached;
  double m_totalAvgNodesReached;
  Vec2D<double> m_calcTimesList;
  Vec1D<double> m_avgCalcTimes;
  std::chrono::time_point<std::chrono::steady_clock> m_startTime;
};

#endif // GAME_H