#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <numeric>
#include <ostream>
#include <sstream>
#include <ranges>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "constants.h"
#include "hashes.h"
#include "types.h"
#include "Population.h"
#include "Species.h"
#include "Network.h"
#include "Random.h"

namespace NEAT
{
    template <
        std::unsigned_integral TInnovation, TInnovation TinCount, TInnovation ToutCount,
        std::default_initializable TActivationFunction,
        std::default_initializable TFitnessFunction, std::default_initializable TNetworkInitializer, std::default_initializable TFinishedEvaluator
    >
    class Simulation
    {
    public:
        using NodeID = InnovationTypes<TInnovation>::NodeID;
        using ConnectionID = InnovationTypes<TInnovation>::ConnectionID;
        using Edge = InnovationTypes<TInnovation>::Edge;

        using Population = Population<TInnovation, TinCount, ToutCount, TActivationFunction>;
        using Species = Population::Species;
        using Network = Species::Network;
        using ConnectionData = Network::ConnectionData;

    public:
        //TODO
        struct TrainingResult
        {
            Population population;
        };


    protected:
        TFitnessFunction evaluateFitness;
        TNetworkInitializer generateInitialNetwork;
        TFinishedEvaluator shouldFinishTraining;

        std::uint32_t populationSize;
        Random random;

        // Compatibility constants
        float c12;
        float c3;
        float deltaT;

        // After [stalePopulationLimit] generations of no fitness improvements
        // in the entire population only [stalePopulationSurvivors] species survive
        std::uint16_t stalePopulationLimit;
        std::uint32_t stalePopulationSurvivors;

        // After [staleSpeciesLimit] generations of no fitness improvements
        // in a species, that species will not produce offspring
        std::uint16_t staleSpeciesLimit;

        // Ratio of offspring of the least performant species in comparison to the most performant one
        float speciesOffspringBaseline;

        // If a species has at least [championSurvivalLimit] members, its
        // most performant member will survive unchanged
        std::uint32_t championSurvivalLimit;

        // Ratio of individuals in each species, which will be discarded before evolution
        float cullFactor;

        // Ratio of offspring of the least performant networks in comparison
        // to the most performant one in each species
        float networkSelectionBaseline;

        // Chance for a weight mutation to occur
        float weightMutateChance;

        // When a weight is being mutated, the chance with which it will be
        // changed slightly instead of reset to a random value
        float perturbChance;

        // Maximum absolute value for weight perturbation
        float perturbAmount;
        
        // Absolute maximum value for fully random weight mutations
        float randomWeightSpread;

        // Chance for a connection enable mutation to occur
        float connectionEnableChance;

        // Chance for a crossover to occur
        float crossoverChance;

        // Chance for interspecies mating to occur
        float interspeciesMatingChance;

        // Chance for an add node mutation to occur
        float addNodeChance;

        // Chance for an add connection mutation to occur
        float addConnectionChance;

        // Maximum absolute value for new connections
        float initialWeightSpread;

    protected:
        // Constructors, Destructors, Methods
        Simulation(
            std::uint32_t populationSize,
            std::uint64_t seed,
            float c12,
            float c3,
            float deltaT,
            std::uint16_t stalePopulationLimit,
            std::uint32_t stalePopulationSurvivors,
            std::uint16_t staleSpeciesLimit,
            float speciesOffspringBaseline,
            std::uint32_t championSurvivalLimit,
            float cullFactor,
            float networkSelectionBaseline,
            float weightMutateChance,
            float perturbChance,
            float perturbAmount,
            float randomWeightSpread,
            float connectionEnableChance,
            float crossoverChance,
            float interspeciesMatingChance,
            float addNodeChance,
            float addConnectionChance,
            float initialWeightSpread
        ) :
            populationSize(populationSize),
            random(seed),
            c12(c12),
            c3(c3),
            deltaT(deltaT),
            stalePopulationLimit(stalePopulationLimit),
            stalePopulationSurvivors(stalePopulationSurvivors),
            staleSpeciesLimit(staleSpeciesLimit),
            speciesOffspringBaseline(speciesOffspringBaseline),
            championSurvivalLimit(championSurvivalLimit),
            cullFactor(cullFactor),
            networkSelectionBaseline(networkSelectionBaseline),
            weightMutateChance(weightMutateChance),
            perturbChance(perturbChance),
            perturbAmount(perturbAmount),
            randomWeightSpread(randomWeightSpread),
            connectionEnableChance(connectionEnableChance),
            crossoverChance(crossoverChance),
            interspeciesMatingChance(interspeciesMatingChance),
            addNodeChance(addNodeChance),
            addConnectionChance(addConnectionChance),
            initialWeightSpread(initialWeightSpread)
        {
            assert(stalePopulationLimit > 0);
            assert(stalePopulationSurvivors >= 2);
            assert(staleSpeciesLimit > 0);
            assert(championSurvivalLimit > 1);
            assert(cullFactor >= 0.0f && cullFactor <= 1.0f);
            assert(weightMutateChance > 0.0f);
            assert(perturbChance > 0.0f);
            assert(connectionEnableChance > 0.0f);
            assert(crossoverChance > 0.0f && crossoverChance <= 1.0f);
            assert(interspeciesMatingChance > 0.0f);
            assert(addNodeChance > 0.0f);
            assert(addConnectionChance > 0.0f);
        }

    public:
        TrainingResult train()
        {
            GenerationCount generation = 0;

            Population p = initializePopulation();

            while (!shouldFinishTraining({ generation++, p }))
            {
                p = p.evolve(
                    populationSize,
                    random,
                    c12,
                    c3,
                    deltaT,
                    stalePopulationLimit,
                    stalePopulationSurvivors,
                    staleSpeciesLimit,
                    [this]() { return initializePopulation(); },
                    championSurvivalLimit,
                    cullFactor,
                    addConnectionChance,
                    addNodeChance,
                    weightMutateChance,
                    connectionEnableChance,
                    crossoverChance,
                    perturbChance,
                    perturbAmount,
                    interspeciesMatingChance,
                    initialWeightSpread,
                    randomWeightSpread
                );
                p.evaluateFitness(evaluateFitness, networkSelectionBaseline, speciesOffspringBaseline);
            }

            return { p };
        }



        Population initializePopulation()
        {
            Population p;

            for (int i = 0; i < populationSize; ++i)
            {
                Species s;
                p.insertNetwork(
                    s,
                    generateInitialNetwork(
                        random,
                        p.getNodeInnovationManager(),
                        p.getConnectionInnovationManager()
                    ),
                    c12,
                    c3,
                    deltaT
                );
            }

            /*
            p.nodeInnovations.clear(); // TODO why was this here???????????
            p.edgeInnovations.clear();
            */

            p.evaluateFitness(
                evaluateFitness,
                networkSelectionBaseline,
                speciesOffspringBaseline
            );

            return p;
        }
    };

    template <
        std::unsigned_integral TInnovation, TInnovation TinCount, TInnovation ToutCount,
        std::default_initializable TActivationFunction
    >
    struct Types // TODO rename
    {
        template <
            std::default_initializable TFitnessFunction,
            std::default_initializable TNetworkInitializer,
            std::default_initializable TFinishedEvaluator
        > using Simulation = Simulation<TInnovation, TinCount, ToutCount, TActivationFunction, TFitnessFunction, TNetworkInitializer, TFinishedEvaluator>;
        using Population = Population<TInnovation, TinCount, ToutCount, TActivationFunction>;
        using Species = Population::Species;
        using Network = Species::Network;

        using NodeID = InnovationTypes<TInnovation>::NodeID;
        using ConnectionID = InnovationTypes<TInnovation>::ConnectionID;
        using Edge = InnovationTypes<TInnovation>::Edge;
    };
}