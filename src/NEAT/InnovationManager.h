#pragma once

#include <concepts>
#include <unordered_map>

namespace NEAT
{


    template <typename T, std::unsigned_integral TInnovation>
    class InnovationManager
    {
    public:
        InnovationManager(TInnovation maxID = 0) : maxID(maxID) { }

        inline TInnovation getInnovation(const T& key)
        {
            return innovations.contains(key) ? innovations[key] : innovations[key] = maxID++;
        }

    private:
        TInnovation maxID;
        std::unordered_map<T, TInnovation> innovations;
    };


}