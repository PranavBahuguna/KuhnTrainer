#include <vector>

template<typename T>
using Vec1D = std::vector<T>;
template<typename T>
using Vec2D = Vec1D<Vec1D<T>>;
template<typename T>
using Vec3D = Vec1D<Vec2D<T>>;
template<typename T>
using Vec4D = Vec1D<Vec3D<T>>;
