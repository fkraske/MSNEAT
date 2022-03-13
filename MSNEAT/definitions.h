#pragma once

#include <cstdint>
#include <unordered_map>

#include "ConnectionSpecialization.h"

namespace Neat
{
	template <std::unsigned_integral TNode>
	using Connection = std::pair<TNode, TNode>;

	template <std::unsigned_integral TNode>
	using ConnectionMap = std::unordered_map<TNode, std::vector<TNode>>;

	template <std::unsigned_integral TNode>
	using SpecializationMap = std::unordered_map<Connection<TNode>, ConnectionSpecialization>;
}