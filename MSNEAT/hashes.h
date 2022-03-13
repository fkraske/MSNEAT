#pragma once

#include <functional>

template<typename T>
struct std::hash<std::pair<T, T>>
{
	std::size_t operator()(const std::pair<T, T>& p) const noexcept
	{
		return std::hash<T>()(p.first) ^ (std::hash<T>()(p.second) << 1);
	}
};