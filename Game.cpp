#include "Game.h"

#include <algorithm>
#include <boost/assign.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <unordered_set>

constexpr bool   USE_TREE_PRUNING  = true;
constexpr double PRUNING_THRESHOLD = 1e-3;
constexpr size_t NUM_THREADS       = 4;
constexpr size_t NUM_ACTIONS       = 2;

template <typename T>
constexpr auto to_underlying(T e) noexcept {
  return static_cast<std::underlying_type_t<T>>(e);
}

/* Constructor
 * @param numPlayers :: Number of players in the game (2 or 3)
 */
Game::Game(const int nPlayers)
    : NUM_PLAYERS(nPlayers),
      NUM_CARDS(nPlayers == 2 ? 3 : 4),
      NUM_LEVELS(nPlayers == 2 ? 4 : 6),
      m_gameCards(nPlayers),
      m_dNodes(NUM_CARDS),
      m_isDNode(NUM_LEVELS),
      m_utilities(NUM_LEVELS),
      m_weights(NUM_LEVELS),
      m_blacklist(NUM_LEVELS - 1) {

  m_rng.seed(std::random_device()());

  // Setup the decision nodes vector
  Vec1D<char> cards = {'J', 'Q', 'K', 'A'};
  for (size_t i = 0; i < NUM_CARDS; ++i) {
    std::string c = std::string(1, cards[i]);

    if (NUM_PLAYERS == 2) {
      m_dNodes[i] = { 
        {DNode(c)},
        {DNode(c + "p"), DNode(c + "b")},
        {DNode(c + "pb")}
      };
    } else {
      m_dNodes[i] = {
        {DNode(c)},
        {DNode(c + "p"), DNode(c + "b")},
        {DNode(c + "pp"), DNode(c + "pb"), DNode(c + "bp"), DNode(c + "bb")},
        {DNode(c + "ppb"), DNode(c + "pbp"), DNode(c + "pbb")},
        {DNode(c + "ppbp"), DNode(c + "ppbb")}
      };
    }
  }

  // Setup decision node indexes vector
  if (NUM_PLAYERS == 2)
    m_dNodeIndexes = {{0}, {0, 1}, {1}, {}};
  else
    m_dNodeIndexes = {{0}, {0, 1}, {0, 1, 2, 3}, {1, 2, 3}, {0, 1}, {}};

  // Setup the decision node check vector
  m_isDNode[0] = {{true}};
  for (size_t level = 0; level < NUM_LEVELS - 1; ++level) {
    size_t numDNodes = m_dNodes[0][level].size();

    m_isDNode[level + 1] = Vec1D<bool>(numDNodes * NUM_ACTIONS);
    for (int index : m_dNodeIndexes[level + 1])
      m_isDNode[level + 1][index] = true;
  }

  // Setup the utilities, terminal utilities, blacklist and weight vectors
  size_t numOutcomes = static_cast<size_t>(boost::math::factorial<double>(
      static_cast<unsigned int>(NUM_PLAYERS)) * NUM_CARDS);
  m_terminalUtilities = Vec4D<double>(numOutcomes, Vec3D<double>(NUM_LEVELS));

  m_utilities[0] = Vec2D<double>(1, Vec1D<double>(NUM_PLAYERS));
  // Top-level node has weights of 1 for each player
  m_weights[0] = Vec2D<double>(1, Vec1D<double>(NUM_PLAYERS, 1.0));

  for (size_t i = 0; i < NUM_LEVELS - 1; ++i) {
    size_t numDNodes = m_dNodes[0][i].size();

    // There is a utility set for each action for each dNode in the next level
    m_utilities[i + 1] = Vec2D<double>(numDNodes * NUM_ACTIONS, Vec1D<double>(NUM_PLAYERS));
    m_weights[i + 1]   = Vec2D<double>(numDNodes * NUM_ACTIONS, Vec1D<double>(NUM_PLAYERS));
    m_blacklist[i]     = Vec1D<bool>(numDNodes * NUM_ACTIONS);

    for (size_t j = 0; j < m_terminalUtilities.size(); ++j)
      m_terminalUtilities[j][i + 1] = Vec2D<double>(numDNodes * NUM_ACTIONS, Vec1D<double>(NUM_PLAYERS));
  }

  // Calculate utilities for all terminal nodes for each possible set of cards
  for (auto &cards : findCardPermutations(NUM_PLAYERS)) {
    size_t comboIndex = calcCardSetIndex(cards);
    
    for (size_t level = 0; level < NUM_LEVELS - 1; ++level) {
      size_t nextLevel = level + 1;
      size_t card = to_underlying(cards[level % NUM_PLAYERS]);

      for (size_t i = 0; i < m_dNodeIndexes[level].size(); ++i) {
        auto &node = m_dNodes[card][level][i];

        for (size_t action = 0; action < NUM_ACTIONS; ++action) {
          size_t nlIndex = i * NUM_ACTIONS + action;

          if (!m_isDNode[nextLevel][nlIndex]) {
            // Node reached by action is terminal - calculate terminal utilities
            std::string terminalHistory = node.getName() + (action == 0 ? "p" : "b");
            calcTerminalUtilities(
                m_terminalUtilities[comboIndex][nextLevel][nlIndex], cards,
                terminalHistory);
          }
        }
      }
    }
  }
}


/* Initialise and resets game parameters
 */
void Game::init() {
  // Initialise all decision nodes
  for (auto &card : m_dNodes) {
    for (auto &level : card) {
      for (auto &dNode : level)
        dNode.init();
    }
  }

  // Add another game values entry for each player
  m_gameValues.push_back(Vec2D<double>(NUM_PLAYERS));
  m_gameValuesPtr = &m_gameValues.back();

  // Reset number of nodes touched
  m_currentNodesReached = 0;

  // Record algorithm start time
  m_startTime = std::chrono::high_resolution_clock::now();
}


/* Train two players against each other for a number of iterations.
 * @param iterations :: The number of iterations to train players
 */
void Game::train(size_t iterations) {

  Vec1D<double> nodesReached(iterations);
  Vec1D<double> calcTimes(iterations);
  Vec1D<double> culGameValues(NUM_PLAYERS);

  for (size_t i = 0; i < iterations; ++i) {

    // Select distinct random cards for each player, then perform CFRM
    selectRandomCards();
    cfrm();

    // Record algorithm stop time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - m_startTime;
    calcTimes[i] = duration.count();

    // Update game values
    const auto &gameValues = m_utilities[0][0];
    for (size_t j = 0; j < NUM_PLAYERS; ++j) {
      culGameValues[j] += gameValues[j];
      m_gameValuesPtr->at(j).push_back(culGameValues[j] / (i + 1.0));
    }

    // Calculate average strategy for each node for this iteration
    for (auto &card : m_dNodes) {
      for (auto &level : card) {
        for (auto &node : level)
          node.calcAvgBetProbabilities();
      }
    }

    // Add new entry for the number of nodes (decision or terminal) touched
    nodesReached[i] = m_currentNodesReached;
  }

  // Add nodes reached counts calculation times to their respective lists
  m_nodesReachedList.push_back(nodesReached);
  m_calcTimesList.push_back(calcTimes);
}


/* Calculates the following properties for this game after running over many
 * iterations / trials:
 * - average bet probabilities for each dNode
 * - average bet probability errors for each dNode
 * - average game values for each player
 * - average game value errors for each player
 * - game value distances from expected values
 *
 * @param numSamples :: The number of samples to take for each property
 */
void Game::calcProperties(size_t numSamples) {
  const size_t numSets = m_gameValues.size();
  const size_t numValues = m_gameValues.front().front().size();
  const size_t sampleStep = numValues / numSamples;

  // Calculate average bet probabilities and errors
  for (auto &card : m_dNodes) {
    for (auto &level : card) {
      for (auto &node : level) {
        auto avgBetProbabilities = node.getAvgBetProbabilities();
        auto avgBetErrors = node.getAvgBetErrors();

        m_avgBetProbabilities.push_back(Vec1D<double>(numSamples));
        m_avgBetErrors.push_back(Vec1D<double>(numSamples));

        for (size_t i = 0; i < numSamples; ++i) {
          size_t sampleIndex = i * sampleStep;
          m_avgBetProbabilities.back()[i] = avgBetProbabilities[sampleIndex];
          m_avgBetErrors.back()[i] = avgBetErrors[sampleIndex];
        }
      }
    }
  }

  m_avgGameValues = Vec2D<double>(NUM_PLAYERS, Vec1D<double>(numSamples));
  m_avgGameValueErrors = Vec2D<double>(NUM_PLAYERS, Vec1D<double>(numSamples));

  // Calculate average game values and errors
  for (size_t i = 0; i < NUM_PLAYERS; ++i) {
    for (size_t j = 0; j < numSamples; ++j) {
      size_t sampleIndex = j * sampleStep;
      double normalizingSum = 0.0;
      double squareSum = 0.0;

      for (const auto &valueSet : m_gameValues)
        normalizingSum += valueSet[i][sampleIndex];
      m_avgGameValues[i][j] = normalizingSum / numSets;

      for (const auto &valueSet : m_gameValues)
        squareSum += pow(valueSet[i][sampleIndex] - m_avgGameValues[i][j], 2.0);
      m_avgGameValueErrors[i][j] = sqrt(squareSum / numSets);
    }
  }

  // Calculate the expected game values and the max game value distances
  if (NUM_PLAYERS == 2) {
    m_expectedGameValues = {-1.0 / 18.0, 1.0 / 18.0};
  } else {
    double handProb = 1.0 / 24.0;
    double jp = m_avgBetProbabilities[1].back();
    double qp = m_avgBetProbabilities[13].back();
    double alpha = std::max(jp, qp);
    m_expectedGameValues = {-handProb * (0.5 + alpha), -handProb * 0.5,
                            handProb * (1.0 + alpha)};
  }

  m_gameValueDists = Vec1D<double>(numSamples);
  for (size_t i = 0; i < numSamples; ++i) {
    size_t sampleIndex = i * sampleStep;
    double maxDist = 0.0;

    for (size_t j = 0; j < NUM_PLAYERS; ++j)
      maxDist = std::max(
          std::abs(m_avgGameValues[j][i] - m_expectedGameValues[j]), maxDist);
    m_gameValueDists[i] = maxDist;
  }

  // Calculate the average number of nodes reached and calculation times in each
  // iteration
  m_avgNodesReached = Vec1D<double>(numSamples);
  m_avgCalcTimes = Vec1D<double>(numSamples);
  for (size_t i = 0; i < numSamples; ++i) {
    size_t sampleIndex = i * sampleStep;
    double nodesReachedSum = 0.0;
    double timesSum = 0.0;

    for (size_t j = 0; j < numSets; ++j) {
      nodesReachedSum += m_nodesReachedList[j][sampleIndex];
      timesSum += m_calcTimesList[j][sampleIndex];
    }
    m_avgNodesReached[i] = nodesReachedSum / numSets;
    m_avgCalcTimes[i] = timesSum / numSets;
  }

  double nodesReachedSum = 0.0;
  for (size_t i = 0; i < numSets; ++i)
    nodesReachedSum += m_nodesReachedList[i].back();

  m_totalAvgNodesReached = nodesReachedSum / numSets;
}


/* Iteratively performs the CounterFactual Regret Minimization algorithm
 */
void Game::cfrm() {

  // Reset the node action blacklist
  for (auto &x : m_blacklist)
    std::fill(x.begin(), x.end(), false);

  size_t cardSetIndex = calcCardSetIndex(m_gameCards);

  // Phase 1 - Iterate downwards through each level of the game tree
  // and calculate the strategies and weights (reach probabilities) for every
  // decision node on each level

  m_currentNodesReached++; // Top-level node always visited
  for (int level = 0; level < NUM_LEVELS - 1; ++level) {
    const int nextLevel = level + 1;
    const int pIndex    = level % NUM_PLAYERS;
    const int card      = to_underlying(m_gameCards[pIndex]);

    // #pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t i = 0; i < m_dNodeIndexes[level].size(); ++i) {
      const int dNodeIndex = m_dNodeIndexes[level][i];
      auto &node = m_dNodes[card][level][i];

      node.calcStrategy(m_weights[level][dNodeIndex][pIndex]);
      const auto &strategy = node.getStrategy();

      for (size_t action = 0; action < NUM_ACTIONS; ++action) {
        size_t nlIndex = i * NUM_ACTIONS + action;
        double betProb = node.getLastAvgBetProb();
        double actionProb = action == to_underlying(Action::P) ? 1 - betProb : betProb;

        // Check if the action from this node is blacklisted - if not, check if
        // it needs blacklisting
        if (USE_TREE_PRUNING &&
            (actionProb < PRUNING_THRESHOLD ||
             (level > 0 && m_blacklist[level - 1][dNodeIndex]))) {
          // Do not visit if the probability of visiting via this action is less
          // than threshold
          m_blacklist[level][nlIndex] = true;
          continue;
        }

        // #pragma omp atomic
        m_currentNodesReached++;

        if (m_isDNode[nextLevel][nlIndex]) {
          // Node reached by action is not terminal - calculate weights
          m_weights[nextLevel][nlIndex] = m_weights[level][dNodeIndex];
          m_weights[nextLevel][nlIndex][pIndex] *= strategy[action];
        }
      }
    }
  }

  // Phase 2 - Iterate upwards through each level of the game tree
  // and calculate the resultant utilities for every decision node
  // on each level

  for (int level = static_cast<int>(NUM_LEVELS - 2); level >= 0; --level) {
    const size_t nextLevel = level + 1;
    const size_t pIndex    = level % NUM_PLAYERS;
    const size_t card      = to_underlying(m_gameCards[pIndex]);

    // #pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t i = 0; i < m_dNodeIndexes[level].size(); ++i) {
      int dNodeIndex = m_dNodeIndexes[level][i];

      // If parent action leading to this node is blacklisted, skip all actions
      if (level > 0 && m_blacklist[level - 1][dNodeIndex])
        continue;

      auto &dNode = m_dNodes[card][level][i];
      const auto &strategy = dNode.getStrategy();

      Vec1D<double> nodeUtilities(NUM_PLAYERS);

      for (size_t action = 0; action < NUM_ACTIONS; ++action) {
        const size_t nlIndex = i * NUM_ACTIONS + action;

        // Skip action if blacklisted
        if (m_blacklist[level][nlIndex])
          continue;

        // Obtain utilities for each player for the given action
        for (size_t player = 0; player < NUM_PLAYERS; ++player) {
          bool isDNode = m_isDNode[nextLevel][nlIndex];
          double playerUtility =
              isDNode ? m_utilities[nextLevel][nlIndex][player]
                      : m_terminalUtilities[cardSetIndex][nextLevel][nlIndex][player];
          nodeUtilities[player] += strategy[action] * playerUtility;
        }
      }

      // Calculate weights derived from other players
      double otherWeights = 1.0;
      for (size_t p = 1; p < NUM_PLAYERS; ++p)
        otherWeights *= m_weights[level][dNodeIndex][(pIndex + p) % NUM_PLAYERS];

      // Update regrets for each action
      for (size_t action = 0; action < NUM_ACTIONS; ++action) {
        const size_t nlIndex = i * NUM_ACTIONS + action;

        bool isDNode = m_isDNode[nextLevel][nlIndex];
        double regret =
            (isDNode ? m_utilities[nextLevel][nlIndex][pIndex]
                     : m_terminalUtilities[cardSetIndex][nextLevel][nlIndex][pIndex]) - 
          nodeUtilities[pIndex];

        double weightedRegret = otherWeights * regret;
        dNode.updateRegretSum(action, weightedRegret);
      }

      // Set utilities at current node to the calculated utilities
      m_utilities[level][dNodeIndex] = nodeUtilities;
    }
  }
}


/* Calculate utilities at terminal nodes
 * @param utilities :: Vector of utilities to calculate for
 * @param cards :: The cards held at this terminal node
 * @param history :: The history of actions taken to reach this node
 */
void Game::calcTerminalUtilities(Vec1D<double> &utilities,
                                 const Vec1D<Card> &cards,
                                 const std::string &history) const {
  // Determine which player 'won' (gained positive utility) and return the
  // utility of this terminal node
  calcPlayerInvestments(utilities, history);
  const double pot = -std::accumulate(utilities.begin(), utilities.end(), 0.0);

  // Determine which players are contesting this terminal node
  Vec1D<size_t> contestants;
  for (size_t i = 0; i < NUM_PLAYERS; ++i) {
    for (size_t j = i + 1; j < history.size(); j += NUM_PLAYERS) {
      if (history[j] == 'b') {
        contestants.push_back(i);
        break;
      }
    }
  }

  // If no contestants found, then all players must be included in the showdown
  if (!contestants.size()) {
    contestants = Vec1D<size_t>(NUM_PLAYERS);
    std::iota(contestants.begin(), contestants.end(), 0);
  }

  // Determine the winner and update their respective utilities
  size_t winner = showdown(cards, contestants);
  utilities[winner] += pot;
}


/* Get the amount of chips each player has invested into a game at a node given
 * its history
 * @param history :: The history preceding the current node, given as string
 * @return :: Vector of coin amounts each player invested
 */
void Game::calcPlayerInvestments(Vec1D<double> &investments,
                                 const std::string &history) const {
  std::fill(investments.begin(), investments.end(), -1.0); // all player ante 1
  for (size_t i = 1; i < history.size(); ++i) {
    size_t pIndex = (i - 1) % NUM_PLAYERS;
    if (history[i] == 'b')
      investments[pIndex]--;
  }
}


/* Obtains the names of all decision nodes used in the game tree
 * @return :: Decision node names, in card, level and index order
 */
Vec1D<std::string> Game::getDNodeNames() const {
  Vec1D<std::string> dNodeNames;

  for (auto &card : m_dNodes) {
    for (auto &level : card) {
      for (auto &node : level)
        dNodeNames.push_back(node.getName());
    }
  }
  return dNodeNames;
}


/* Select a number distinct random cards out of the available cards
 */
void Game::selectRandomCards() {
  const std::uniform_real_distribution<double> dist(0.0, static_cast<double>(NUM_CARDS));

  std::unordered_set<size_t> randIndexes;
  while (randIndexes.size() < static_cast<size_t>(NUM_CARDS - 1)) {
    // Select random indexes, reject and retry with ones we have already found
    size_t randIndex = static_cast<size_t>(std::floor(dist(m_rng)));

    if (!randIndexes.count(randIndex))
      randIndexes.emplace(randIndex);
  }

  // Convert random indexes into cards
  std::transform(randIndexes.begin(), randIndexes.end(), m_gameCards.begin(),
                 [](size_t i) -> Card { return Card(i); });
}


/* Selects a random action according to a strategy
 * @param strategy :: The probabilities of selecting each action, summing to 1
 */
Action Game::selectRandomAction(const Vec1D<double> &strategy) {
  // Select random number between 0 and 1
  const std::uniform_real_distribution<double> dist(0.0, 1.0);
  double rand = std::floor(dist(m_rng));

  size_t i = 0;
  double probabilitySum = strategy.front();
  while (rand >= probabilitySum)
    probabilitySum += strategy[++i];

  return Action(i);
}


/* Determine the winner of a card showdown for a given list of participating
 * players
 * @param cards :: The cards held by each player
 * @param playerIndexes :: Indexes of the participating players
 * @return :: Index of the winning player
 */
size_t Game::showdown(const Vec1D<Card> &cards,
                      const Vec1D<size_t> &playerIndexes) const {
  return *std::max_element(playerIndexes.begin(), playerIndexes.end(),
                           [cards](const size_t &p1, const size_t &p2) {
                             return cards[p1] < cards[p2];
                           });
}


/* Discovers all possible permutations of every card combination for a given
 * number of cards
 * @param k :: The size of each permutation set
 * @return :: All possible card permutations
 */
Vec2D<Card> Game::findCardPermutations(size_t k) const {
  Vec2D<Card> combinations;
  Vec2D<Card> permutations;

  findCardCombinations(combinations, Vec1D<Card>(NUM_PLAYERS), k, 0, 0, NUM_CARDS - 1);

  for (auto &combination : combinations) {
    do {
      permutations.push_back(combination);
    } while (std::next_permutation(combination.begin(), combination.end()));
  }

  return permutations;
}


/* Recursively discovers all possible combinations of cards for a given number
 * of cards
 * @param combinations :: Vector of every discovered combination
 * @param currentCombination :: The current card combination being built
 * @param k :: The size of each combination set
 * @param index :: The index within the card combination set
 * @param start :: The index to start construction from
 * @param end :: The index to end construction at
 */
void Game::findCardCombinations(Vec2D<Card> &combinations,
                                Vec1D<Card> &currentCombination,
                                size_t k,
                                size_t index,
                                size_t start,
                                size_t end) const {
  // Combination ready, add to the permutations list
  if (index == k) {
    combinations.push_back(currentCombination);
    return;
  }

  // Replace index with all possible elements
  for (size_t i = start; i <= end && end - i + 1 >= k - index; ++i) {
    currentCombination[index] = Card(i);
    findCardCombinations(combinations, currentCombination, k, index + 1, i + 1, end);
  }
}


/* Calculates a unique, consecutive index for a given card set
 * @param cards :: The cards to find an index for
 * @return :: The calculated index
 */
size_t Game::calcCardSetIndex(const Vec1D<Card> &cards) const {
  size_t a = to_underlying(cards[0]);
  size_t b = to_underlying(cards[1]);
  if (NUM_PLAYERS == 2) {
    size_t A = 2 * a;
    size_t B = b + ((b < a) ? 1 : 0) - 1;
    return A + B;
  } else {
    size_t c = to_underlying(cards[2]);
    size_t A = 6 * a;
    size_t B = 2 * (b + ((b < a) ? 1 : 0) - 1);
    size_t C = (c < std::min(a, b) ? c : (c < std::max(a, b) ? c - 1 : c - 2));
    return A + B + C;
  }
}


/* Output the names and average strategy of each decision node
 */
std::ostream &operator<<(std::ostream &os, const Game &game) {
  for (auto &card : game.m_dNodes) {
    for (auto &level : card) {
      for (auto &node : level) {
        os << std::setw(8) << node.getName() << ": " << node << std::endl;
      }
    }
  }
  os << std::endl;

  // Calculate and print player game values, expected game values and the
  // epsilon-Nash value
  const auto &avgGameValues = game.getAvgGameValues();
  const auto &expGameValues = game.getExpectedGameValues();
  for (size_t i = 0; i < game.NUM_PLAYERS; ++i) {
    double difference = std::abs(avgGameValues[i].back() - expGameValues[i]);
    os << "Player " << i + 1 << " game value = " << std::setprecision(6)
       << avgGameValues[i].back() << ", expected value = " << expGameValues[i]
       << std::setprecision(6) << ", difference = " << difference
       << std::setprecision(6) << std::endl;
  }

  os << std::endl
     << "e-Nash = " << game.getGameValueDists().back() << std::setprecision(6)
     << std::endl;
  os << "Nodes reached = " << game.getTotalAvgNodesReached() << std::endl;
  os << "Avg. calculation time = " << game.getAvgCalcTimes().back() << "s"
     << std::endl;

  return os;
}