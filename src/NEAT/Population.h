#pragma once

#include <algorithm>
#include <ranges>

#include "Species.h"
#include "InnovationManager.h"
#include "util.h"

namespace NEAT
{


    template <
        std::unsigned_integral TInnovation, TInnovation TinCount, TInnovation ToutCount,
        std::default_initializable TActivationFunction
    >
    struct Population
    {
        struct Snapshot
        {
            GenerationCount generation;
            Population<TInnovation, TinCount, ToutCount, TActivationFunction>& population;
        };

        using Species = Species<TInnovation, TinCount, ToutCount, TActivationFunction>;
        using Network = Species::Network;

        using NodeID = InnovationTypes<TInnovation>::NodeID;
        using ConnectionID = InnovationTypes<TInnovation>::ConnectionID;
        using Edge = InnovationTypes<TInnovation>::Edge;
        using NodeInnovationManager = InnovationTypes<TInnovation>::NodeInnovationManager;
        using ConnectionInnovationManager = InnovationTypes<TInnovation>::ConnectionInnovationManager;

        std::vector<Species> species;

        NodeInnovationManager nodeInnovationManager{ ToutCount + TinCount + 1 };
        ConnectionInnovationManager connectionInnovationManager;

        float combinedSpeciesWeight = 0;
        Fitness highestFitness = MINIMUM_FITNESS;
        Fitness previousHighestFitness = MINIMUM_FITNESS;
        Fitness highestSpeciesAdjustedFitness = MINIMUM_FITNESS;
        Fitness lowestSpeciesAdjustedFitness = MAXIMUM_FITNESS;

        std::uint16_t staleGenerations = 0;

        template <typename TPopulationInitializer>
        Population evolve(
            std::uint32_t populationSize,
            Random& random,
            float c12,
            float c3,
            float deltaT,
            std::uint16_t stalePopulationLimit,
            std::uint32_t stalePopulationSurvivors,
            std::uint16_t staleSpeciesLimit,
            TPopulationInitializer initializePopulation,
            std::uint32_t championSurvivalLimit,
            float cullFactor,
            float addConnectionChance,
            float addNodeChance,
            float weightMutateChance,
            float connectionEnableChance,
            float crossoverChance,
            float perturbChance,
            float perturbAmount,
            float interspeciesMatingChance,
            float initialWeightSpread,
            float randomWeightSpread
        )
        {
            Population pN;
            pN.previousHighestFitness = highestFitness;

            /* THIS */
            pN.nodeInnovationManager = nodeInnovationManager;
            pN.connectionInnovationManager = connectionInnovationManager;
            /* WAS PREVIOUSLY THIS *\
             * pN.setMaxNodeID(getMaxNodeID());
             * pN.setMaxConnectionID(getMaxConnectionID());
            \* */ // TODO check if still (or now?) correct

            if (staleGenerations >= stalePopulationLimit)
            {
                // Retain only most performant Species if Population is stale
                std::ranges::sort(
                    species,
                    [](const Species& first, const Species& second)
                    {
                        return second.combinedAdjustedFitness < first.combinedAdjustedFitness;
                    }
                );

                species.resize(std::clamp<size_t>(stalePopulationSurvivors, 1, species.size()));

                // TODO fix offspring count in this case
            }
            else
            {
                // Remove stale Species
                species.resize(
                    std::remove_if(
                        species.begin(),
                        species.end(),
                        [this, staleSpeciesLimit](const Species& s) { return s.staleGenerations >= staleSpeciesLimit; }
                    ) - species.begin()
                );
                // TODO fix offspring count in this case
                // TODO this can remove all species at once

                if (species.empty())
                    *this = initializePopulation();
            }

            for (Species& s : species)
                s.remainingOffspring = std::max(
                    1u,
                    random.round<std::uint32_t>(
                        combinedSpeciesWeight
                        ? populationSize * s.weight / combinedSpeciesWeight
                        : static_cast<float>(populationSize) / species.size()
                    )
                );

            //TODO try out culling population-wide instead of per species
            // Cull each Species' weakest individuals
            if (cullFactor > 0)
                for (Species& s : species)
                {
                    std::ranges::sort(s.networks, [](const Network& first, const Network& second) { return second.adjustedFitness < first.adjustedFitness; });
                    s.networks.resize(std::clamp<size_t>(s.networks.size() * cullFactor, 1, s.networks.size()));
                }

            // Insert previous species
            // NOTE: This initializes each kept Species with a temporary individual (its former champion)
            for (Species& s : species)
                if (s.remainingOffspring)
                    pN.carryOverSpecies(s);

            // Champions survive
            for (Species& s : species)
                if (s.networks.size() >= championSurvivalLimit)
                    pN.insertNetwork(s, s.networks.front(), c12, c3, deltaT);

            // Interspecies mating
            // TODO the chance should probably apply per species or crossover not per population
            // TODO this does not reduce the remaining offspring of any species
            // TODO all combinedSpeciesWeight == 0 checks should be able to be eliminated since it should never be 0
            random.repeatFractional(
                [this, &random, &pN, c12, c3, deltaT]()
                {
                    float l1 = random.getFloat();

                    auto itB = species.begin();

                    for (auto it1 = itB; it1 != species.end(); ++it1)
                    {
                        float rP1 = combinedSpeciesWeight
                            ? it1->weight / combinedSpeciesWeight
                            : 1.0f / species.size();

                        if (l1 < rP1)
                        {
                            float l2 = random.getFloat() - rP1;

                            for (auto it2 = it1 == itB ? itB + 1 : itB; it2 != species.end(); ++it2 != it1 ? it2 : ++it2)
                            {
                                float rP2 = combinedSpeciesWeight
                                    ? it2->weight / combinedSpeciesWeight
                                    : 1.0f / species.size();

                                if (l2 < rP2)
                                    pN.insertNetwork(
                                        random.getBool() ? *it1 : *it2,
                                        it1->getRandomPerformanceWeightedParent(random)->crossover(
                                            *it2->getRandomPerformanceWeightedParent(random),
                                            random
                                        ),
                                        c12,
                                        c3,
                                        deltaT
                                    );
                                else
                                    l2 -= rP2;
                            }
                        }
                        else
                            l1 -= rP1;
                    }

                },
                interspeciesMatingChance
            );

            // Remaining Offspring Generation
            // TODO crossover excludes other mutations
            // TODO maybe mutations exclude each other in general
            for (Species& s : species)
                while (s.remainingOffspring > 0)
                {
                    auto it = s.getRandomPerformanceWeightedParent(random);
                    Network n = s.networks.size() > 1 && random.getFloat() < crossoverChance ?
                        it->crossover(*s.getRandomPerformanceWeightedParent(random, it), random) :
                        *it;

                    random.repeatFractional(
                        [&n, &random, perturbChance, perturbAmount, randomWeightSpread]() mutable
                        {
                            n = n.mutateWeight(
                                random,
                                perturbChance,
                                perturbAmount,
                                randomWeightSpread
                            );
                        },
                        weightMutateChance
                    );
                    random.repeatFractional(
                        [&n, &random]() mutable
                        {
                            n = n.mutateEnableConnection(random);
                        },
                        connectionEnableChance
                    );
                    random.repeatFractional(
                        [&n, &random, &pN]() mutable
                        {
                            n = n.mutateAddNode(
                                random,
                                pN.nodeInnovationManager,
                                pN.connectionInnovationManager
                            );
                        },
                        addNodeChance
                    );
                    random.repeatFractional(
                        [&n, &random, &pN, initialWeightSpread]() mutable
                        {
                            n = n.mutateAddConnection(
                                random,
                                pN.connectionInnovationManager,
                                initialWeightSpread
                            );
                        },
                        addConnectionChance
                    );

                    pN.insertNetwork(s, n, c12, c3, deltaT);
                }

            // Remove Species representatives
            for (auto it = pN.species.begin(); it != pN.species.end();)
            {
                if (it->networks.size() == 1)
                {
                    it = pN.species.erase(it);
                }
                else
                {
                    it->networks.erase(it->networks.begin());
                    ++it;
                }
            }

            return pN;
        }

        // TODO check if correct
        template <typename T>
        void evaluateFitness(
            T& evaluateNetworkFitness,
            float networkSelectionBaseline,
            float speciesOffspringBaseline
        )
        {
            for (Species& s : species)
            {
                for (Network& n : s.networks)
                {
                    highestFitness = std::max(
                        highestFitness,
                        s.highestFitness = std::max(
                            s.highestFitness,
                            n.fitness = evaluateNetworkFitness(n)
                        )
                    );
                    s.lowestFitness = std::min(s.lowestFitness, n.fitness);
                    s.combinedAdjustedFitness += n.adjustedFitness = n.fitness / s.networks.size();
                }

                lowestSpeciesAdjustedFitness = std::min(lowestSpeciesAdjustedFitness, s.combinedAdjustedFitness);
                highestSpeciesAdjustedFitness = std::max(highestSpeciesAdjustedFitness, s.combinedAdjustedFitness);
                s.staleGenerations = s.highestFitness > s.previousHighestFitness ? 0 : s.staleGenerations + 1;
            }

            for (Species& s : species)
            {
                if (s.highestFitness - s.lowestFitness)
                    for (Network& n : s.networks)
                        s.combinedNetworkWeight += n.weight = rescale(
                            n.fitness,
                            s.lowestFitness,
                            s.highestFitness,
                            networkSelectionBaseline * (s.highestFitness - s.lowestFitness),
                            s.highestFitness - s.lowestFitness
                        );

                if (highestSpeciesAdjustedFitness - lowestSpeciesAdjustedFitness)
                    combinedSpeciesWeight += s.weight = rescale(
                        s.combinedAdjustedFitness,
                        lowestSpeciesAdjustedFitness,
                        highestSpeciesAdjustedFitness,
                        speciesOffspringBaseline * (highestSpeciesAdjustedFitness - lowestSpeciesAdjustedFitness),
                        highestSpeciesAdjustedFitness - lowestSpeciesAdjustedFitness
                    );
            }

            staleGenerations = highestFitness > previousHighestFitness ? 0 : staleGenerations + 1;
        }

        void insertNetwork(Species& originSpecies, const Network& network, float c12, float c3, float deltaT)
        {
            --originSpecies.remainingOffspring; // TODO awkward

            for (Species& s : species)
                if (network.isCompatible(s.networks.front(), c12, c3, deltaT))
                {
                    s.networks.push_back(std::move(network));
                    return;
                }

            addNewSpecies(network);
        }

        inline NodeInnovationManager& getNodeInnovationManager()
        {
            return nodeInnovationManager;
        }

        inline ConnectionInnovationManager& getConnectionInnovationManager()
        {
            return connectionInnovationManager;
        }

        inline NodeID getNodeInnovation(ConnectionID connection)
        {
            return nodeInnovationManager.getInnovation(connection);
        }

        inline ConnectionID getConnectionInnovation(Edge edge)
        {
            return connectionInnovationManager.getInnovation(edge);
        }

        inline ConnectionID getConnectionInnovation(NodeID in, NodeID out)
        {
            return getConnectionInnovation({ in, out });
        }

        auto getNetworks()
        {
            return species | std::views::transform([](Species& s) -> std::vector<Network>& { return s.networks; }) | std::views::join;
        }

        Network& getFittest()
        {
            Network* result = &species.front().networks.front();
            // TODO max_element? (doesn't work with ranges for some reason .__.)
            //return *std::ranges::max_element(getNetworks(), [](const Network& first, const Network& second) { return first.fitness < second.fitness; });
            for (Network& n : getNetworks())
                if (n.fitness > result->fitness)
                    result = &n;

            return *result;
        }

        void addNewSpecies(const Network& base)
        {
            species.push_back({ { base, base } }); // TODO what is this??
        }

        void carryOverSpecies(const Species& oldSpecies)
        {
            Species s{ std::vector<Network>{ oldSpecies.networks.front() } };

            s.previousHighestFitness = oldSpecies.highestFitness;
            s.staleGenerations = oldSpecies.staleGenerations;

            species.push_back(std::move(s));
        }

        std::string basicInfo()
        {
            std::stringstream ss;

            ss << "Population Size: " << std::ranges::distance(getNetworks()) << "\n"
                << "Number of Species: " << species.size() << "\n"
                << "Highest Fitness : " << highestFitness << "\n";

            return ss.str();
        }

        std::string speciesInfo()
        {
            std::stringstream ss;

            for (Species& s : species)
                ss << "{\n" << s.basicInfo(1) << "},\n";

            return ss.str();
        }

        std::string networkInfo()
        {
            std::stringstream ss;

            for (Species& s : species)
                ss << s.networkInfo(1) << ",\n";

            return ss.str();
        }
    };


}