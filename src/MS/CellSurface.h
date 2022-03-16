#pragma once

#include <cstdint>

namespace MS
{
    enum class CellSurface : std::uint16_t
    {
        Bomb   = 0,
        Empty  = 1 << 0,
        One    = 1 << 1,
        Two    = 1 << 2,
        Three  = 1 << 3,
        Four   = 1 << 4,
        Five   = 1 << 5,
        Six    = 1 << 6,
        Seven  = 1 << 7,
        Eight  = 1 << 8,
        Hidden = 1 << 9,
    };
}