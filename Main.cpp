#include "Game.h"
#include "GraphPlotter.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

int main() {
  // Init python interpreter and add modules to sys path
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"./PythonModules/\")");

  // Obtain game parameters from input
  int numPlayers;
  std::cout << "Type (2) to play 2-player Kuhn Poker, type (3) to play "
               "3-player Kuhn Poker:"
            << std::endl;
  std::cin >> numPlayers;

  size_t numTrials;
  std::cout << "Enter the number of trials:" << std::endl;
  std::cin >> numTrials;

  size_t numIterations;
  std::cout << "Enter the number of iterations to play per trial:" << std::endl;
  std::cin >> numIterations;

  size_t numSamples;
  std::cout << "Enter the number of samples to use for graphs:" << std::endl;
  std::cin >> numSamples;

  size_t xAxisUnits;
  std::cout << "Choose the units for the x-axis to be displayed in:"
            << std::endl
            << "1 - Iterations" << std::endl
            << "2 - Nodes reached" << std::endl
            << "3 - Calculation times" << std::endl;
  std::cin >> xAxisUnits;

  std::string filename;
  std::cout << "Enter filename:" << std::endl;
  std::cin >> filename;

  if (numTrials < 1 || numIterations < 1)
    return 0;

  // Initialise game and train for a number of iterations
  Game game(numPlayers);
  for (size_t i = 0; i < numTrials; ++i) {
    game.init();
    game.train(numIterations);
  }
  game.calcProperties(numSamples);

  // Print results to stdout
  std::cout << "\nResults:\n--------\n\n" << game;

  GraphPlotter gPlot;
  std::string title = "Average betting probability over time"; // title
  std::string xLabel;                                          // x-axis label
  std::string yLabel = "Avg. betting probability";             // y-axis label

  std::vector<double> xValues(numSamples); // x-values
  switch (xAxisUnits) {
  case 1: {
    int current = 0;
    int step = static_cast<int>(numIterations / numSamples);
    for (auto &x : xValues) {
      x = current;
      current += step;
    }
    xLabel = "Iterations";
    break;
  }
  case 2:
    xValues = game.getAvgNodesReached();
    xLabel = "Nodes reached";
    break;
  case 3:
    xValues = game.getAvgCalcTimes();
    xLabel = "Time (s)";
    break;
  }

  if (numPlayers == 2) {
    const auto &avgBetProb = game.getAvgBetProbabilities(); // y-values
    const auto &avgBetErrors = game.getAvgBetErrors();      // e-values
    const auto &dNodeNames = game.getDNodeNames();          // legend

    // Plot graphs of average betting probability used by each decision node for
    // each number of nodes reached
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        size_t nodeIndex = i * 4 + j;
        gPlot.plot(xValues, avgBetProb[nodeIndex], avgBetErrors[nodeIndex],
                   dNodeNames[nodeIndex]);
      }
      gPlot.showPlot(title, xLabel, yLabel);
    }
  }

  const auto &avgGameValues = game.getAvgGameValues();           // y-values
  const auto &avgGameValueErrors = game.getAvgGameValueErrors(); // e-values
  const auto &playerNames =
      std::vector<std::string>{"Player 1", "Player 2", "Player 3"}; // legend

  // Plot graph of average game value for each number of nodes reached for each
  // player
  for (size_t i = 0; i < numPlayers; ++i)
    gPlot.plot(xValues, avgGameValues[i], avgGameValueErrors[i],
               playerNames[i]);
  gPlot.showPlot(title, xLabel, yLabel, "", "");

  title = "Average game value distance over time";       // title
  yLabel = "Avg. game value distance";                   // y-axis label
  const auto &gameValueDists = game.getGameValueDists(); // y-values
  std::vector<double> errorsVec(numSamples);             // e-values
  std::fill(std::begin(errorsVec), std::end(errorsVec), 0.0);

  // Plot graph of average game value distance from expected for each player
  gPlot.plot(xValues, gameValueDists, errorsVec, "distances");
  gPlot.showPlot(title, xLabel, yLabel, "", "log");

  std::ofstream file;
  file.open(filename, std::ios::app);
  for (auto &x : xValues)
    file << x << ",";
  file << std::endl;
  for (auto &y : gameValueDists)
    file << y << ",";
  file << std::endl;

  Py_Finalize();
  return 0;
}