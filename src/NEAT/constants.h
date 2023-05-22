#pragma once

#include <limits>

#include "types.h"

namespace NEAT
{


    static inline constexpr Fitness MINIMUM_FITNESS = std::numeric_limits<Fitness>::lowest();
    static inline constexpr Fitness MAXIMUM_FITNESS = std::numeric_limits<Fitness>::max();


}