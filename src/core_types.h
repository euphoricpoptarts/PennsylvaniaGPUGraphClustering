#pragma once

#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

typedef int32_t ordinal_t;
typedef int32_t edge_offset_t;
typedef edge_offset_t value_t;

using Device = Kokkos::Cuda;
using matrix_t = typename KokkosSparse::CrsMatrix<value_t, ordinal_t, Device, void, edge_offset_t>;