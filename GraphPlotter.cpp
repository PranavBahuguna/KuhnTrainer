#include "GraphPlotter.h"

// Constructor
GraphPlotter::GraphPlotter() {
  // Try to import the GraphPlot python module
  PyObject *gPlot_module = PyImport_ImportModule("GraphPlot");
  if (gPlot_module == NULL) {
    PyErr_Print();
    printf("Error in loading python module, plotting disabled.");
  } else {
    // Obtain all functions from module
    PyObject *gplot_dict = PyModule_GetDict(gPlot_module);
    m_plot_func = PyDict_GetItemString(gplot_dict, "plot");
    m_showPlot_func = PyDict_GetItemString(gplot_dict, "showPlot");
  }
}

// Destructor
GraphPlotter::~GraphPlotter() { Py_Finalize(); }

/* Plots an errorbar graph
 * @param x :: x-values vector
 * @param y :: y-values vector
 * @param e :: errors for y-values vector
 * @param label :: the label to assign to this graph
 */
void GraphPlotter::plot(const std::vector<double> &x,
                        const std::vector<double> &y,
                        const std::vector<double> &e,
                        const std::string &label) const {
  if (!PyCallable_Check(m_plot_func))
    return;

  PyObject *pyXVec = vecToPyTuple(x);
  PyObject *pyYVec = vecToPyTuple(y);
  PyObject *pyEVec = vecToPyTuple(e);
  PyObject *pyLabel = PyUnicode_FromString(label.c_str());
  PyObject *args = PyTuple_Pack(4, pyXVec, pyYVec, pyEVec, pyLabel);
  PyObject_CallObject(m_plot_func, args);
}

/* Show plot with given labels
 * @param title :: title of the graph
 * @param xLabel :: x-axis label
 * @param yLabel :: y-axis label
 * @param yScaling :: scaling type to use for x-axis
 * @param yScaling :: scaling type to use for y-axis
 */
void GraphPlotter::showPlot(const std::string &title, const std::string &xLabel,
                            const std::string &yLabel,
                            const std::string &xScaling,
                            const std::string &yScaling) const {
  if (!PyCallable_Check(m_showPlot_func))
    return;

  PyObject *pyTitle = PyUnicode_FromString(title.c_str());
  PyObject *pyXLabel = PyUnicode_FromString(xLabel.c_str());
  PyObject *pyYLabel = PyUnicode_FromString(yLabel.c_str());
  PyObject *pyXScaling = PyUnicode_FromString(xScaling.c_str());
  PyObject *pyYScaling = PyUnicode_FromString(yScaling.c_str());
  PyObject *args =
      PyTuple_Pack(5, pyTitle, pyXLabel, pyYLabel, pyXScaling, pyYScaling);
  PyObject_CallObject(m_showPlot_func, args);
}

/* Converts a C++ vector of doubles into a Python tuple
 */
PyObject *GraphPlotter::vecToPyTuple(const std::vector<double> &vec) const {
  PyObject *pyVecTuple = PyTuple_New(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    PyObject *pyVal = PyFloat_FromDouble(vec[i]);
    PyTuple_SetItem(pyVecTuple, i, pyVal);
  }
  return pyVecTuple;
}