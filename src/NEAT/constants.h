#pragma once

#include <limits>

#include "types.h"

namespace NEAT
{


    static inline constexpr Fitness minimumFitness = std::numeric_limits<Fitness>::lowest();
    static inline constexpr Fitness maximumFitness = std::numeric_limits<Fitness>::max();


}