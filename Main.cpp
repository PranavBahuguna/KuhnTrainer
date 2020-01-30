#include "Game.h"
#include "GraphPlotter.h"

#include <boost/lexical_cast.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#define XAXIS_ITERATIONS    1
#define XAXIS_NODES_REACHED 2
#define XAXIS_CALC_TIMES    3

int main(int argc, char *argv[]) {
  // Init python interpreter and add modules to sys path
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"./PythonModules/\")");

  // Obtain game parameters from input
  size_t nPlayers    = boost::lexical_cast<size_t>(argv[0]);
  size_t nTrials     = boost::lexical_cast<size_t>(argv[1]);
  size_t nIterations = boost::lexical_cast<size_t>(argv[2]);
  size_t nSamples    = boost::lexical_cast<size_t>(argv[3]);
  size_t xAxisUnits  = boost::lexical_cast<size_t>(argv[4]);

  // Initialise game and train for a number of iterations
  Game game(nPlayers);
  for (size_t i = 0; i < nTrials; ++i) {
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

  std::vector<double> xValues(nSamples); // x-values

  // Fill xValues according to the x-axis units selection
  switch (xAxisUnits) {
    case XAXIS_ITERATIONS: {
      // [DEVELOPMENT]: Replace this with some kind of generator function
      double current = 0;
      double step = nIterations / nSamples;
      for (auto &x : xValues) {
        x = current;
        current += step;
      }
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
      std::vector<std::string>{"Player 1", "Player 2", "Player 3"}; // legend

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

  // Plot graph of average game value distance from expected for each player
  gPlot.plot(xValues, gameValueDists, errorsVec, "distances");
  gPlot.showPlot(title, xLabel, yLabel, "", "log");

  Py_Finalize();
  return 0;
}