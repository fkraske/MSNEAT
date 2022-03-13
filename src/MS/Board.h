#pragma once

#include <cstdint>
#include <utility>

#include "CellSurface.h"

namespace MS
{
    template <size_t Twidth, size_t Theight, std::uint32_t Tmines>
    class Board
    {
    private:
        std::array<bool, Twidth * Theight> mines;
        std::array<bool, Twidth * Theight> revealed;
        std::array<std::uint8_t, Twidth * Theight> numbers;

    public:
        Board(
            const std::array<bool, Twidth * Theight>& mines,
            const std::array<bool, Twidth * Theight>& revealed = { }
        ) : mines(mines), revealed(revealed)
        {
            for (int i = 0; i < mines.size(); ++i)
                if (mines[i])
                    for (int j = -Twidth; j <= Twidth; j += Twidth)
                        for (int k = -1; k <= 1; ++k)
                            if (size_t c = i + j + k; c >= 0 && c < Twidth * Theight)
                                ++numbers[c];
        }

        CellSurface getSurface(size_t i)
        {
            return revealed ? static_cast<CellSurface>(1 << getNumber(i)) : CellSurface::Hidden;
        }

        inline CellSurface getSurface(size_t x, size_t y)
        {
            getSurface(index2DtoIndex1D(x, y));
        }

        inline std::uint8_t getNumber(size_t i)
        {
            return numbers[i];
        }

        inline std::uint8_t getNumber(size_t x, size_t y)
        {
            return getNumber(index2DtoIndex1D(x, y));
        }

        inline size_t index2DtoIndex1D(size_t x, size_t y)
        {
            return x + y * Twidth;
        }

        inline std::pair<size_t, size_t> index1DtoIndex2D(size_t i)
        {
            return { i % Twidth, i / Twidth };
        }
    };

    using EasyBoard = Board<10, 10, 10>;
    using IntermediateBoard = Board<16, 16, 40>;
    using ExpertBoard = Board<30, 16, 99>;
}