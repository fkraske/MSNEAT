#pragma once

#include <type_traits>

#include "types.h"

namespace NEAT
{


    template <typename T>
    concept Action = std::is_invocable_r_v<void, T>;


}