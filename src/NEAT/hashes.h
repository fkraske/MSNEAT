#pragma once

#include <functional>
#include <utility>

//TODO check if this is correct
template <typename T1, typename T2>
struct std::hash<std::pair<T1, T2>>
{
    size_t operator()(const std::pair<T1, T2>& pair) const
    {
        static std::hash<T1> h1;
        static std::hash<T2> h2;
        return h1(pair.first) ^ (h2(pair.second) << 1u);
    }
};