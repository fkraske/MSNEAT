#pragma once

#include <cstdint>
#include <ostream>
#include <utility>

#include "CellSurface.h"

namespace MS
{
    template <size_t Twidth, size_t Theight, std::uint32_t Tmines>
    class Board
    {
    public:
        friend std::ostream;

    private:
        std::array<bool, Twidth * Theight> mines;
        std::array<bool, Twidth * Theight> revealed;
        std::array<std::uint8_t, Twidth * Theight> numbers;

    public:
        Board(
            const std::array<bool, Twidth * Theight>& mines,
            const std::array<bool, Twidth * Theight>& revealed = { }
        ) : mines(mines), revealed(revealed), numbers()
        {
            for (size_t j = 0; j < Theight; ++j)
                for (size_t i = 0; i < Twidth; ++i)
                    if (mines[index2DtoIndex1D(i, j)])
                        for (size_t l = std::max(0uLL, j - 1); l <= std::min(Theight - 1, j + 1); ++l)
                            for (size_t k = std::max(0uLL, i - 1); k <= std::min(Twidth - 1, i + 1); ++k)
                                ++numbers[index2DtoIndex1D(k, l)];
        }

        CellSurface getSurface(size_t i) const
        {
            return revealed[i] ? mines[i] ? CellSurface::Bomb : static_cast<CellSurface>(1 << getNumber(i)) : CellSurface::Hidden;
        }

        inline CellSurface getSurface(size_t x, size_t y) const
        {
            return getSurface(index2DtoIndex1D(x, y));
        }

        inline std::uint8_t getNumber(size_t i) const
        {
            return numbers[i];
        }

        inline std::uint8_t getNumber(size_t x, size_t y) const
        {
            return getNumber(index2DtoIndex1D(x, y));
        }

        inline static size_t index2DtoIndex1D(size_t x, size_t y)
        {
            return x + y * Twidth;
        }

        inline static std::pair<size_t, size_t> index1DtoIndex2D(size_t i)
        {
            return { i % Twidth, i / Twidth };
        }
    };

    using EasyBoard = Board<10, 10, 10>;
    using IntermediateBoard = Board<16, 16, 40>;
    using ExpertBoard = Board<30, 16, 99>;
}

template <size_t Twidth, size_t Theight, std::uint32_t Tmines>
std::ostream& operator<<(std::ostream& os, const MS::Board<Twidth, Theight, Tmines>& board)
{
    for (size_t j = 0; j < Theight; ++j)
    {
        for (size_t i = 0; i < Twidth; ++i)
        {
            auto s = board.getSurface(i, j);

            switch (s)
            {
            case MS::CellSurface::Empty:
                os << " ";
                break;
            case MS::CellSurface::Bomb:
                os << "X";
                break;
            case MS::CellSurface::Hidden:
                os << "-";
                break;
            default:
                os << static_cast<std::uint32_t>(std::log2(static_cast<std::uint16_t>(s)));
                break;
            }

            if (i < Twidth - 1)
                os << " ";
        }

        if (j < Theight - 1)
            os << "\n";
    }

    return os;
}