#pragma once

#include <functional>
#include <utility>

//TODO check if this is correct
template <typename T1, typename T2>
struct std::hash<std::pair<T1, T2>>
{
    size_t operator()(const std::pair<T1, T2>& pair) const
    {
        return std::hash<T1>()(pair.first) ^ (std::hash<T2>()(pair.second) << 1u);
    }
};