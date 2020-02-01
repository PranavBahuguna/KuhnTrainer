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

template<typename T>
using Vec1D = std::vector<T>;

template<typename T>
using Vec2D = Vec1D<Vec1D<T>>;

template<typename T>
using Vec3D = Vec1D<Vec2D<T>>;

template<typename T>
using Vec4D = Vec1D<Vec3D<T>>;

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
  std::vector<std::string> getDNodeNames() const;

  friend std::ostream &operator<<(std::ostream &os, const Game &game);

private:
  Vec2D<Card> findCardPermutations(size_t k) const;
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

  const int NUM_PLAYERS;
  const int NUM_CARDS;
  const int NUM_LEVELS;

  std::mt19937 m_rng;
  Vec1D<Card> m_gameCards;
  Vec3D<double> m_gameValues;
  Vec2D<double> *m_gameValuesPtr;

  Vec3D<DecisionNode> m_dNodes;
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