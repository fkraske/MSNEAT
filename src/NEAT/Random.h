#pragma once

#include <concepts>
#include <cstdint>
#include <random>

#include "concepts.h"

namespace NEAT
{


    class Random
    {
    public:
        Random(std::uint64_t seed);

        inline bool getBool()
        {
            return getUInt64(2);
        }

        inline float getFloat()
        {
            return realDistribution(rng);
        }

        inline std::uint64_t getUInt64()
        {
            return rng();
        }

        inline std::uint64_t getUInt64(std::uint64_t max)
        {
            return std::uniform_int_distribution<std::uint64_t>(0, max - 1)(rng);
        }

        template <std::unsigned_integral TUI, std::floating_point TF>
        inline TUI round(TF f)
        {
            TUI base = static_cast<TUI>(f);

            return base + (getFloat() < (f - base));
        }

        template <Action T>
        inline void repeatFractional(T&& f, float times) // TODO probably doesn't work with function pointer
        {
            for (; times > 1; --times)
                f();

            if (getFloat() < times)
                f();
        }

        template <Action T>
        inline void repeat(T&& f, float minTimes, float maxTimes)
        {
            float remainingTimes = minTimes + getFloat() * (maxTimes - minTimes);

            for (; remainingTimes > 1; --remainingTimes)
                f();

            if (getFloat() < remainingTimes)
                f();
        }

    private:
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> realDistribution;
    };


}