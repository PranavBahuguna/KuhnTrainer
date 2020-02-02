#include "DataTypes.h"
#include "Game.h"
#include "GraphPlotter.h"

#include <boost/lexical_cast.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

constexpr int XAXIS_ITERATIONS    = 0;
constexpr int XAXIS_NODES_REACHED = 1;
constexpr int XAXIS_CALC_TIMES    = 2;

int main(int argc, char *argv[]) {
  // Init python interpreter and add modules to sys path
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"./PythonModules/\")");

  // Obtain game parameters from input
  int nPlayers    = boost::lexical_cast<int>(argv[1]);
  int nTrials     = boost::lexical_cast<int>(argv[2]);
  int nIterations = boost::lexical_cast<int>(argv[3]);
  int nSamples    = boost::lexical_cast<int>(argv[4]);
  int xAxisUnits  = boost::lexical_cast<int>(argv[5]);
  
  // Initialise game and train for a number of iterations
  Game game(nPlayers);
  for (int i = 0; i < nTrials; ++i) {
    game.init();
    game.train(nIterations);
  }
  game.calcProperties(nSamples);

  // Print results to stdout
  std::cout << "\nResults:\n--------\n\n" << game;

  GraphPlotter gPlot;
  std::string title = "Average betting probability over time"; // title
  std::string xLabel;                                          // x-axis label
  std::string yLabel = "Avg. betting probability";             // y-axis label
  Vec1D<double> xValues(nSamples);                             // x-values

  // Fill xValues according to the x-axis units selection
  switch (xAxisUnits) {
    case XAXIS_ITERATIONS: {
      double step = nIterations / nSamples;
      std::generate(xValues.begin(), xValues.end(), [i = 0.0, step]() mutable { return i += step; });
      xLabel = "Iterations";
      break;
    }
    
    case XAXIS_NODES_REACHED:
      xValues = game.getAvgNodesReached();
      xLabel = "Nodes reached";
      break;

    case XAXIS_CALC_TIMES:
      xValues = game.getAvgCalcTimes();
      xLabel = "Time (s)";
      break;
  }

  if (nPlayers == 2) {
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

  title = "Average game value over time";                           // title
  yLabel = "Avg. game value";                                       // y-axis label
  const auto &avgGameValues = game.getAvgGameValues();              // y-values
  const auto &avgGameValueErrors = game.getAvgGameValueErrors();    // e-values
  const auto &playerNames =
      Vec1D<std::string>{"Player 1", "Player 2", "Player 3"};       // legend

  // Plot graph of average game value for each number of nodes reached for each
  // player
  for (size_t i = 0; i < nPlayers; ++i)
    gPlot.plot(xValues, avgGameValues[i], avgGameValueErrors[i], playerNames[i]);

  gPlot.showPlot(title, xLabel, yLabel, "", "");

  title = "Average e-Nash value over time";                         // title
  yLabel = "Avg. e-Nash value";                                     // y-axis label
  const auto &gameValueDists = game.getGameValueDists();            // y-values
  std::vector<double> errorsVec(nSamples);                          // e-values
  std::fill(std::begin(errorsVec), std::end(errorsVec), 0.0);

  // Plot graph of average game value distance (e-Nash) from expected for each player
  gPlot.plot(xValues, gameValueDists, errorsVec, "");
  gPlot.showPlot(title, xLabel, yLabel, "", "log");

  Py_Finalize();
  return 0;
}