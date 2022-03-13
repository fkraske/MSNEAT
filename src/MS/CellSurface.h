#pragma once

#include <cstdint>

namespace MS
{
    enum class CellSurface : std::uint16_t
    {
        Empty  = 0,
        One    = 1,
        Two    = 1 << 1,
        Three  = 1 << 2,
        Four   = 1 << 3,
        Five   = 1 << 4,
        Six    = 1 << 5,
        Seven  = 1 << 6,
        Eight  = 1 << 7,
        Hidden = 1 << 8,
        Bomb   = 1 << 9,
    };
}