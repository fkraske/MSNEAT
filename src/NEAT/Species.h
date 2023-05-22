#pragma once

#include "Network.h"

namespace NEAT
{


    template <
        std::unsigned_integral TInnovation, TInnovation TinCount, TInnovation ToutCount,
        std::default_initializable TActivationFunction
    >
    struct Species
    {
        using Network = Network<TInnovation, TinCount, ToutCount, TActivationFunction>;

        std::vector<Network> networks;

        Fitness combinedAdjustedFitness = MINIMUM_FITNESS;
        float combinedNetworkWeight = 0;
        float weight = 0;
        Fitness highestFitness = MINIMUM_FITNESS;
        Fitness previousHighestFitness = MINIMUM_FITNESS;
        Fitness lowestFitness = MAXIMUM_FITNESS;

        std::uint32_t remainingOffspring = 0;
        std::uint16_t staleGenerations = 0;

        auto getRandomPerformanceWeightedParent(Random& random) const
        {
            float r = random.getFloat() * combinedNetworkWeight;

            for (auto it = networks.begin(); it != networks.end(); ++it)
            {
                float weight = it->weight;

                if (r < weight)
                    return it;
                else
                    r -= weight;
            }

            return networks.end() - 1;
        }

        auto getRandomPerformanceWeightedParent(Random& random, std::vector<Network>::const_iterator ignore) const
        {
            float r = random.getFloat() * combinedNetworkWeight - ignore->weight;

            auto itB = networks.begin();

            for (auto it = itB == ignore ? itB + 1 : itB; it != networks.end(); ++it != ignore ? it : ++it)
            {
                float weight = it->weight;

                if (r < weight)
                    return it;
                else
                    r -= weight;
            }

            return networks.end() - 1;
        }

        std::string basicInfo(int indent = 0)
        {
            std::stringstream ss;

            ss << std::string(indent, '\t') << "Number of Networks: " << networks.size() << "\n"
                << std::string(indent, '\t') << "Highest Fitness : " << highestFitness << "\n";

            return ss.str();
        }

        std::string networkInfo(int indent = 0)
        {
            std::stringstream ss;

            ss << std::string(indent, '\t') << "{\n";

            for (Network& n : networks)
                ss << n.detailedInfo(indent + 1);

            ss << std::string(indent, '\t') << "}";

            return ss.str();
        }
    };


}