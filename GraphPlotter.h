#ifndef GRAPHPLOTTER_H
#define GRAPHPLOTTER_H

#include <Python.h>

#include <vector>
#include <string>

class GraphPlotter {

public:
  GraphPlotter();
  ~GraphPlotter();
  void plot(const std::vector<double> &x, const std::vector<double> &y,
            const std::vector<double> &e, const std::string &label) const;
  void showPlot(const std::string &title, const std::string &xLabel,
                const std::string &yLabel, const std::string &xScaling = "",
                const std::string &yScaling = "") const;

private:
  PyObject *vecToPyTuple(const std::vector<double> &vec) const;
  PyObject *m_plot_func;
  PyObject *m_showPlot_func;
};

#endif // GRAPHPLOTTER_H