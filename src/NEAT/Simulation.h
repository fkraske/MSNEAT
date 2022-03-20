#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <numeric>
#include <ostream>
#include <sstream>
#include <random>
#include <ranges>
#include <unordered_set>
#include <unordered_map>
#include <vector>

//TODO remove
#include <iostream>

#include "hashes.h"

//TODO make performant :P
//TODO test stuff
//TODO maybe add fitness and generation count types as template parameters
namespace Neat
{
    template <std::unsigned_integral TInnovation, TInnovation TinCount, TInnovation ToutCount>
    class Simulation
    {
        // Types
    public:
        using NodeID = TInnovation;
        using ConnectionID = TInnovation;
        using Edge = std::pair<NodeID, NodeID>;
        using GenerationCount = std::uint64_t;
        using Fitness = float;

        //TODO
        static inline constexpr TInnovation inputSize = TinCount;
        static inline constexpr TInnovation outputSize = ToutCount;
        static inline constexpr Fitness minimumFitness = std::numeric_limits<Fitness>::lowest();
        static inline constexpr Fitness maximumFitness = std::numeric_limits<Fitness>::max();

        struct ConnectionData
        {
            TInnovation innovation;
            float weight;
            bool enabled;
        };

        using ConnectionMap = std::unordered_map<NodeID, std::unordered_map<NodeID, ConnectionData>>;

        struct Network;

        struct Species
        {
            std::vector<Network> networks;

            Fitness combinedAdjustedFitness = minimumFitness;
            float combinedNetworkWeight = 0;
            float weight = 0;
            Fitness highestFitness = minimumFitness;
            Fitness previousHighestFitness = minimumFitness;
            Fitness lowestFitness = maximumFitness;

            std::uint32_t remainingOffspring = 0;

            std::uint16_t staleGenerations = 0;

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

        struct Population
        {
            std::vector<Species> species;

            NodeID maxNodeID = outputSize + inputSize + 1;
            ConnectionID maxConnectionID = 0;
            std::unordered_map<ConnectionID, NodeID> nodeInnovations;
            std::unordered_map<Edge, ConnectionID> edgeInnovations;

            float combinedSpeciesWeight = 0;
            Fitness highestFitness = minimumFitness;
            Fitness previousHighestFitness = minimumFitness;
            Fitness highestSpeciesAdjustedFitness = minimumFitness;
            Fitness lowestSpeciesAdjustedFitness = maximumFitness;

            std::uint16_t staleGenerations = 0;

            inline NodeID getNodeInnovation(ConnectionID connection)
            {
                return nodeInnovations.contains(connection) ? nodeInnovations[connection] : nodeInnovations[connection] = maxNodeID++;
            }

            inline ConnectionID getConnectionInnovation(Edge edge)
            {
                return edgeInnovations.contains(edge) ? edgeInnovations[edge] : edgeInnovations[edge] = maxConnectionID++;
            }

            inline ConnectionID getConnectionInnovation(NodeID in, NodeID out)
            {
                return getConnectionInnovation(Edge(in, out));
            }

            //TODO what does this do?
            auto getNetworks()
            {
                return species | std::views::transform([](const auto& s) { return s.networks; }) | std::views::join;
            }

            Network& getFittest()
            {
                Network* result = &species.front().networks.front();

                for (Species& s : species)
                    for (Network& n : s.networks)
                        if (n.fitness > result->fitness)
                            result = &n;

                return *result;
            }

            void addNewSpecies(const Network& base)
            {
                species.push_back({ { base, base } });
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

                ss  << "Population Size: " << std::ranges::distance(getNetworks()) << "\n"
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

        struct Network
        {
            std::vector<NodeID> nodes;
            ConnectionMap connections;
            Fitness fitness = minimumFitness;
            Fitness adjustedFitness = minimumFitness;
            float weight = 0;

            inline ConnectionData& getConnectionData(NodeID in, NodeID out)
            {
                return connections[out][in];
            }

            inline ConnectionData& getConnectionData(const Edge& edge)
            {
                return getConnectionData(edge.first, edge.second);
            }

            inline const ConnectionData& getConnectionData(NodeID in, NodeID out) const
            {
                return connections.at(out).at(in);
            }

            inline const ConnectionData& getConnectionData(const Edge& edge) const
            {
                return getConnectionData(edge.first, edge.second);
            }

            inline bool containsEdge(NodeID in, NodeID out) const
            {
                return connections.contains(out) && connections.at(out).contains(in);
            }

            inline bool containsEdge(const Edge& edge) const
            {
                return containsEdge(edge.first, edge.second);
            }

            inline bool containsEdgeWithInnovation(NodeID in, NodeID out, ConnectionID innovation) const
            {
                return containsEdge(in, out) && getConnectionData(in, out).innovation == innovation;
            }

            inline bool containsEdgeWithInnovation(const Edge& edge, ConnectionID innovation) const
            {
                return containsEdgeWithInnovation(edge.first, edge.second, innovation);
            }

            bool containsIndirectReverseConnection(
                std::vector<NodeID>::reverse_iterator inIt,
                std::vector<NodeID>::reverse_iterator outIt
            ) const
            {
                for (auto it = inIt + 1; it < outIt; ++it)
                    if (containsEdge(*it, *inIt))
                        return containsIndirectReverseConnection(it, outIt);

                return containsEdge(*outIt, *inIt);
            }

            auto getEdges()
            {
                return connections | std::views::transform(
                    [](const auto& p1) {
                        const auto& [out, connected] = p1;

                        return connected | std::views::transform(
                            [&out](const auto& p2) {
                                const auto& [in, _] = p2;

                                return Edge(in, out);
                            });
                    })
                    | std::views::join;
            }

            const auto getEdges() const
            {
                return connections | std::views::transform(
                    [](const auto& p1) {
                        const auto& [out, connected] = p1;

                        return connected | std::views::transform(
                            [&out](const auto& p2) {
                                const auto& [in, _] = p2;

                                return Edge(in, out);
                            });
                    })
                    | std::views::join;
            }

            std::string nodeInfo(int indent = 0)
            {
                std::stringstream ss;

                ss << std::string(indent, '\t') << "{ ";

                for (NodeID n : nodes)
                    ss << n << ", ";

                ss << "}";

                return ss.str();
            }

            std::string connectionInfo(int indent = 0)
            {
                std::stringstream ss;

                ss << std::string(indent, '\t') << "{\n";

                for (auto e : getEdges())
                {
                    const ConnectionData& data = getConnectionData(e);

                    ss << std::string(indent + 1, '\t') << "{ "
                        << e.first << ", "
                        << e.second << ": "
                        << data.innovation << ", "
                        << data.weight << ", "
                        << data.enabled << " }, \n";
                }

                ss << std::string(indent, '\t') << "}";

                return ss.str();
            }

            std::string detailedInfo(int indent = 0)
            {
                std::stringstream ss;

                ss << std::string(indent, '\t') << "{\n";
                ss << std::string(indent + 1, '\t') << "Fitness: " << fitness << ",\n";
                ss << std::string(indent + 1, '\t') << "Nodes:\n" << nodeInfo(indent + 1) << ",\n";
                ss << std::string(indent + 1, '\t') << "Edges:\n" << connectionInfo(indent + 1) << ",\n";
                ss << std::string(indent, '\t') << "},\n";

                return ss.str();
            }
        };

        struct Snapshot
        {
            GenerationCount generation;
            Population& population;
        };

    public:
        //TODO
        struct TrainingResult
        {
            Population population;
        };


        // Fields
    protected:
        std::uint32_t populationSize;
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> realDistribution;
        float c12;
        float c3;
        float deltaT;
        std::uint16_t stalePopulationLimit;
        std::uint32_t stalePopulationSurvivors;
        std::uint16_t staleSpeciesLimit;
        float speciesOffspringBaseline;
        std::uint32_t championSurvivalLimit;
        float cullFactor;
        float networkSelectionBaseline;
        float weightMutateChance;
        float perturbChance;
        float perturbAmount;
        float randomWeightSpread;
        float connectionEnableChance;
        float crossoverChance;
        float interspeciesMatingChance;
        float addNodeChance;
        float addConnectionChance;
        float initialWeightSpread;

        // Constructors, Destructors, Methods
    protected:
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
            rng(seed),
            realDistribution(0.0f, 1.0f),
            c12(c12),
            c3(),
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

        virtual float activation(float x) const = 0;
        virtual Fitness evaluateFitness(const Network& network, NodeID maxNodeID) = 0;
        virtual bool shouldFinishTraining(Snapshot snapshot) = 0;
        virtual Population initializePopulation() = 0;

    public:
        TrainingResult train()
        {
            GenerationCount generation = 0;

            Population p = initializePopulation();

            do
            {
                // EVOLUTION

                Population pN;
                pN.previousHighestFitness = p.highestFitness;
                pN.maxNodeID = p.maxNodeID;
                pN.maxConnectionID = p.maxConnectionID;

                if (p.staleGenerations >= stalePopulationLimit)
                {
                    // Retain only most performant Species if Population is stale
                    std::ranges::sort(
                        p.species,
                        [](const Species& first, const Species& second)
                        {
                            return second.combinedAdjustedFitness - first.combinedAdjustedFitness;
                        }
                    );

                    p.species.resize(std::clamp<size_t>(stalePopulationSurvivors, 1, p.species.size()));

                    //TODO fix offspring count in this case
                }
                else
                {
                    // Remove stale Species
                    p.species.resize(
                        std::remove_if(
                            p.species.begin(),
                            p.species.end(),
                            [this](const Species& s) { return s.staleGenerations >= staleSpeciesLimit; }
                        ) - p.species.begin()
                    );
                    //TODO fix offspring count in this case

                    if (p.species.empty())
                        p = initializePopulation();
                }

                //TODO make new using def for population values

                for (Species& s : p.species)
                    s.remainingOffspring = std::max(
                        1u,
                        randomRound<std::uint32_t>(
                            p.combinedSpeciesWeight
                                ? populationSize * s.weight / p.combinedSpeciesWeight
                                : static_cast<float>(populationSize) / p.species.size()
                            )
                    );

                //TODO try out culling population-wide instead of per species
                // Cull each Species' weakest individuals
                if (cullFactor > 0)
                {
                    for (Species& s : p.species)
                    {
                        std::ranges::sort(s.networks, [](const Network& first, const Network& second) { return second.adjustedFitness - first.adjustedFitness; });
                        s.networks.resize(std::clamp<size_t>(s.networks.size() * cullFactor, 1, s.networks.size()));
                    }
                }

                // Insert previous species
                for (Species& s : p.species)
                    if (s.remainingOffspring)
                        pN.carryOverSpecies(s);

                // Champions survive
                for (Species& s : p.species)
                    if (s.networks.size() >= championSurvivalLimit)
                        insertNetwork(pN, s, s.networks.front());

                // Interspecies mating
                repeatWithRandomMantissa(
                    [this, &p, &pN]()
                    {
                        float l1 = getRandomFloat();

                        auto itB = p.species.begin();

                        for (auto it1 = itB; it1 != p.species.end(); ++it1)
                        {
                            float rP1 = p.combinedSpeciesWeight
                                ? it1->weight / p.combinedSpeciesWeight
                                : 1 / p.species.size();

                            if (l1 < rP1)
                            {
                                float l2 = getRandomFloat() - rP1;

                                for (auto it2 = it1 == itB ? itB + 1 : itB ; it2 != p.species.end(); ++it2 != it1 ? it2 : ++it2)
                                {
                                    float rP2 = p.combinedSpeciesWeight
                                        ? it2->weight / p.combinedSpeciesWeight
                                        : 1 / p.species.size();

                                    if (l2 < rP2)
                                        insertNetwork(
                                            pN,
                                            getRandomBool() ? *it1 : *it2,
                                            crossover(
                                                *getRandomPerformanceWeightedParent(*it1),
                                                *getRandomPerformanceWeightedParent(*it2)
                                            )
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
                for (Species& s : p.species)
                    while (s.remainingOffspring > 0)
                    {
                        auto it = getRandomPerformanceWeightedParent(s);
                        Network n = (s.networks.size() > 1 && getRandomFloat() < crossoverChance)
                            ? crossover(*it, *getRandomPerformanceWeightedParent(s, it))
                            : *it;

                        repeatWithRandomMantissa([this, &pN, &n]() { n = mutateAddConnection(pN, n); }, addConnectionChance);
                        repeatWithRandomMantissa([this, &pN, &n]() { n = mutateAddNode(pN, n); }, addNodeChance);
                        repeatWithRandomMantissa([this, &n]() { n = mutateWeight(n); }, weightMutateChance);
                        repeatWithRandomMantissa([this, &n]() { n = mutateEnableConnection(n); }, connectionEnableChance);

                        insertNetwork(pN, s, n);
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

                p = std::move(pN);



                // FITNESS EVALUATION

                for (Species& s : p.species)
                {
                    for (Network& n : s.networks)
                    {
                        p.highestFitness = std::max(
                            p.highestFitness,
                            s.highestFitness = std::max(
                                s.highestFitness,
                                n.fitness = evaluateFitness(n, p.maxNodeID)
                            )
                        );
                        s.lowestFitness = std::min(s.lowestFitness, n.fitness);
                        s.combinedAdjustedFitness += n.adjustedFitness = n.fitness / s.networks.size();
                    }

                    p.lowestSpeciesAdjustedFitness = std::min(p.lowestSpeciesAdjustedFitness, s.combinedAdjustedFitness);
                    p.highestSpeciesAdjustedFitness = std::max(p.highestSpeciesAdjustedFitness, s.combinedAdjustedFitness);
                    s.staleGenerations = s.highestFitness > s.previousHighestFitness ? 0 : s.staleGenerations + 1;
                }

                for (Species& s : p.species)
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

                    if (p.highestSpeciesAdjustedFitness - p.lowestSpeciesAdjustedFitness)
                        p.combinedSpeciesWeight += s.weight = rescale(
                            s.combinedAdjustedFitness,
                            p.lowestSpeciesAdjustedFitness,
                            p.highestSpeciesAdjustedFitness,
                            speciesOffspringBaseline * (p.highestSpeciesAdjustedFitness - p.lowestSpeciesAdjustedFitness),
                            p.highestSpeciesAdjustedFitness - p.lowestSpeciesAdjustedFitness
                        );
                }

                p.staleGenerations = p.highestFitness > p.previousHighestFitness ? 0 : p.staleGenerations + 1;
            } while (!shouldFinishTraining({ generation++, p }));

            return { p }; //TODO
        }



        //Mutations

        Network mutateWeight(const Network& network)
        {
            Network result(network);

            auto edges = result.getEdges();

            if (edges.begin() == edges.end())
                return result;



            ConnectionData& c = result.getConnectionData(getRandomEdge(result));

            c.weight =
                getRandomFloat() < perturbChance
                ? c.weight * (1.0f + (getRandomFloat() - 0.5f) * perturbAmount)
                : 2.0f * (getRandomFloat() - 0.5f) * randomWeightSpread;

            return result;
        }

        Network mutateEnableConnection(const Network& network)
        {
            Network result(network);

            auto disabledEdges = result.getEdges() | std::views::filter([&result](Edge e) { return !result.getConnectionData(e).enabled; });
            auto disabledSize = std::ranges::distance(disabledEdges);

            if (disabledSize > 0)
                result.getConnectionData(*std::ranges::next(disabledEdges.begin(), getRandomInt(disabledSize))).enabled = true;

            return result;
        }

        //TODO don't give up after one try (maybe std::shuffle, to iterate over all possible node pairs?)
        //TODO find out why theres only 1 node, 5 edges and not 1 node, 6/7/8 edges
        Network mutateAddConnection(Population& population, const Network& network)
        {
            Network result(network);

            size_t nodesSize = result.nodes.size();

            size_t inNode = getRandomInt(nodesSize + inputSize + 1);
            size_t outNode = getRandomInt(nodesSize + outputSize - (inNode < nodesSize));

            outNode += inNode < nodesSize && inNode <= outNode;

            if (!result.containsEdge(inNode, outNode))
            {
                if (inNode >= nodesSize || outNode >= nodesSize)
                {
                    inNode = inNode < nodesSize ? result.nodes[inNode] : inNode + outputSize - nodesSize;
                    outNode = outNode < nodesSize ? result.nodes[outNode] : outNode - nodesSize;
                }
                else
                {
                    auto inIt = result.nodes.rbegin() + result.nodes.size() - 1 - inNode;
                    auto outIt = result.nodes.rbegin() + result.nodes.size() - 1 - outNode;

                    if (inIt < outIt)
                        if (network.containsIndirectReverseConnection(inIt, outIt))
                            std::rotate(inIt, inIt + 1, outIt + 1);
                        else
                            return result;

                    inNode = result.nodes[inNode];
                    outNode = result.nodes[outNode];
                }


                result.getConnectionData(inNode, outNode) = {
                    population.getConnectionInnovation(inNode, outNode),
                    (1.0f + (getRandomFloat() - 0.5f)) * initialWeightSpread,
                    true
                };
            }

            return result;
        }

        //TODO same node can be added multiple times in the same generation and same individual
        Network mutateAddNode(Population& population, const Network& network)
        {
            Network result(network);

            auto edges = result.getEdges();

            if (edges.begin() == edges.end())
                return result;



            Edge e = getRandomEdge(result);
            ConnectionData& d = result.connections[e.second][e.first];
            NodeID newNode = population.getNodeInnovation(d.innovation);

            d.enabled = false;

            result.getConnectionData(e.first, newNode) = { population.getConnectionInnovation(newNode, e.first), 1.0f, true };
            result.getConnectionData(newNode, e.second) = { population.getConnectionInnovation(e.second, newNode), d.weight, true };
            result.nodes.insert(std::ranges::find(result.nodes, e.second), newNode);

            return result;
        }

        Network crossover(const Network& first, const Network& second)
        {
            Network result;

            const Network& fitter = first.fitness > second.fitness ? first : second;
            const Network& lessFit = first.fitness <= second.fitness ? first : second;

            result.nodes = fitter.nodes;
            result.connections = fitter.connections;

            auto es = fitter.getEdges();

            for (Edge e : es)
                if (lessFit.containsEdgeWithInnovation(e, fitter.getConnectionData(e).innovation) && getRandomInt(2))
                    result.getConnectionData(e) = lessFit.getConnectionData(e);

            return result;
        }



        inline bool getRandomBool()
        {
            return getRandomInt(2);
        }

        inline float getRandomFloat()
        {
            return realDistribution(rng);
        }

        inline std::uint64_t getRandomInt()
        {
            return rng();
        }

        inline std::uint64_t getRandomInt(std::uint64_t max)
        {
            return getRandomInt() % max;
        }

        template <std::unsigned_integral TUI, std::floating_point TF>
        inline TUI randomRound(TF f)
        {
            TUI base = static_cast<TUI>(f);

            return base + (getRandomFloat() < (f - base));
        }

        template <std::floating_point T>
        inline T rescale(T v, T oldMin, T oldMax, T newMin, T newMax)
        {
            if (oldMax == oldMin)
                return v;

            return (v - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;
        }



        inline std::array<float, outputSize> generateOutput(const Network& network, const std::array<float, inputSize>& input, NodeID maxNodeID) const
        {
            std::array<float, outputSize> finalOutput{ };
            std::vector<float> output(maxNodeID, 0);

            std::ranges::copy(input, output.begin() + outputSize);
            for (int i = outputSize; i < outputSize + inputSize; ++i)
                output[i] = activation(output[i]);

            output[outputSize + inputSize] = 1;

            for (NodeID hidden : network.nodes)
            {
                //TODO use std::ranges::any instead
                bool applyActivation = false;

                for (const auto& [in, data] : network.connections.at(hidden))
                    if (data.enabled)
                    {
                        output[hidden] += output[in] * data.weight;
                        applyActivation = true;
                    }

                if (applyActivation)
                    output[hidden] = activation(output[hidden]);
            }

            for (NodeID out = 0; out < outputSize; ++out)
            {
                //TODO use std::ranges::any instead
                bool applyActivation = false;

                if (network.connections.contains(out))
                    for (const auto& [in, data] : network.connections.at(out))
                        if (data.enabled)
                        {
                            finalOutput[out] += output[in] * data.weight;
                            applyActivation = true;
                        }

                if (applyActivation)
                    finalOutput[out] = activation(finalOutput[out]);
            }

            return finalOutput;
        }
        //TODO make member functions const

        //TODO can't i fit this in the respective classes?
        void insertNetwork(Population& population, Species& originSpecies, const Network& network)
        {
            --originSpecies.remainingOffspring;

            for (Species& s : population.species)
                if (isCompatible(network, s.networks.front()))
                {
                    s.networks.push_back(std::move(network));
                    return;
                }

            population.addNewSpecies(network);
        }

        inline bool isCompatible(const Network& first, const Network& second) const
        {
            auto flattened = first.getEdges();
            auto otherFlattened = second.getEdges();
            auto matching = flattened | std::views::filter([&second](const Edge& e) { return second.containsEdge(e); });

            auto flattenedSize = std::ranges::distance(flattened);
            auto otherFlattenedSize = std::ranges::distance(otherFlattened);
            auto matchingSize = std::ranges::distance(matching);

            auto ED = flattenedSize + otherFlattenedSize - 2 * matchingSize;
            auto N = std::max(flattenedSize, otherFlattenedSize);

            if (!N)
                return true;

            if (!matchingSize)
                return false;

            float W = 0.0f;

            for (Edge e : matching)
                W += std::abs(second.getConnectionData(e).weight - first.getConnectionData(e).weight);

            W /= matchingSize;

            return c12 * ED / N + c3 * W <= deltaT;
        }

        inline auto getRandomPerformanceWeightedParent(const Species& species)
        {
            float r = getRandomFloat();

            for (auto it = species.networks.begin(); it != species.networks.end(); ++it)
            {
                float weight = species.combinedNetworkWeight
                    ? it->weight / species.combinedNetworkWeight
                    : 1.0f / species.networks.size();

                if (r < weight)
                    return it;
                else
                    r -= weight;
            }

            return species.networks.end() - 1;
        }

        inline auto getRandomPerformanceWeightedParent(const Species& species, std::vector<Network>::const_iterator ignore)
        {
            float r = getRandomFloat() - (species.combinedNetworkWeight ? (ignore->weight / species.combinedNetworkWeight) : 1.0f / species.networks.size());

            auto itB = species.networks.begin();

            for (auto it = itB == ignore ? itB + 1 : itB; it != species.networks.end(); ++it != ignore ? it : ++it)
            {
                float weight = species.combinedNetworkWeight
                    ? it->weight / species.combinedNetworkWeight
                    : 1.0f / species.networks.size();

                if (r < weight)
                    return it;
                else
                    r -= weight;
            }

            return species.networks.end() - 1;
        }

        inline void repeatWithRandomMantissa(std::function<void()> f, float times)
        {
            for (; times > 1; --times)
                f();

            if (getRandomFloat() < times)
                f();
        }

        inline void repeatRandomly(std::function<void()> f, float maxTimes, float minTimes = 0)
        {
            float remainingTimes = minTimes + getRandomFloat() * (maxTimes - minTimes);

            for (; remainingTimes > 1; --remainingTimes)
                f();

            if (getRandomFloat() < remainingTimes)
                f();
        }

        inline Edge getRandomEdge(Network& network)
        {
            auto flattened = network.getEdges();

            return *std::ranges::next(flattened.begin(), getRandomInt(std::ranges::distance(flattened)));
        }
    };
}