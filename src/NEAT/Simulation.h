#pragma once

#include <array>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>

#include "definitions.h"
#include "hashes.h"

//TODO bias (additional input node with constant 1 input)
namespace Neat
{
    template <size_t TinCount, size_t ToutCount, std::unsigned_integral TNode>
    class Simulation
    {
        // Types
    protected:
        using GenerationCount = std::uint64_t;
        //TODO is this necessary?
        using Fitness = float;

        //TODO maybe convert to using definition
        struct Genome
        {
            SpecializationMap<TNode> genes;
            Fitness fitness;
        };

        struct Species
        {
            std::vector<Genome> genomes;

            std::vector<TNode> hiddenNodes;
            std::vector<Connection<TNode>> connections;
        };

        struct Population
        {
            TNode maxNode;
            ConnectionMap<TNode> connections;

            std::vector<Species> species;

            Population() : maxNode(), connections(), species() { }

            Population(TNode maxNode) :
                maxNode(maxNode),
                connections(),
                species({ Species{ std::vector(maxNode, Genome{ }) } })
            { }
        };

        struct Snapshot
        {
            GenerationCount generation;
            Population& population;
        };

        struct NetworkConfiguration
        {
            const std::vector<TNode>& speciesHiddenNodes;
            const ConnectionMap<TNode>& populationConnections;
            const SpecializationMap<TNode>& genomeGenes;
        };

    public:
        //TODO
        struct TrainingResult
        {

        };


        // Fields
    private:
        std::uint32_t populationSize;
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> realDistribution;
        float perturbChance;
        float perturbAmount;
        float deltaT;
        float c12;
        float c3;

        // Constructors, Destructors, Methods
    protected:
        Simulation(
            std::uint32_t populationSize,
            std::uint64_t seed,
            float perturbChance,
            float perturbAmount,
            float deltaT,
            float c12,
            float c3
        ) :
            populationSize(populationSize),
            rng(seed),
            realDistribution(0.0f, 1.0f),
            perturbChance(perturbChance),
            perturbAmount(perturbAmount),
            deltaT(deltaT),
            c12(c12),
            c3()
        { }

        virtual std::array<float, TinCount> getNetworkInput() = 0;
        virtual float activation(float f) = 0;
        virtual Fitness evaluateFitness(const NetworkConfiguration& network) = 0;
        virtual bool trainingFinishCondition(const Snapshot& snapshot) = 0;

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

    public:
        TrainingResult train()
        {
            GenerationCount generation = 1;
            auto pp = std::make_unique<Population>(Population{ ToutCount + TinCount + 1 });

            while (true) {

                Population& p = *pp;

                //Fitness calculation
                Snapshot snapshot{ generation, p };

                for (auto& s : p.species)
                    for (auto& g : s.genomes)
                        g.fitness = evaluateFitness({ s.hiddenNodes, p.connections, g.genes }) / s.genomes.size();

                //Loop Condition
                if (trainingFinishCondition(snapshot))
                    return { }; //TODO

                ++generation;

                //Evolution
                Population pN;

                for (const auto& s : p.species)
                {

                }
                //Speciation
                //Mutation (Weights, Enabled/Disable, Nodes, Connections)
                //Crossover
                //Culling

                pp = std::make_unique<Population>(pN);
            }

            return { }; //TODO
        }

        Genome mutateWeight(const Genome& genome)
        {
            Genome result(genome);
            
            auto it = (result.genes.begin() + getRandomInt(result.genes.size()));

            it->weight =
                getRandomFloat() < perturbChance
                ? it->weight * (1.0f + (getRandomFloat() - 0.5f) * perturbAmount)
                : 2.0f * (getRandomFloat() - 0.5f);

            return result;
        }

        Genome mutateAddConnection(const Genome& genome)
        {
            Genome result(genome);

            //TODO

            return result;
        }

        Genome mutateAddNode(const Genome& genome)
        {
            Genome result(genome);

            //TODO

            return result;
        }

        //TODO maybe swap cycle and Neat-based compatibility check and just make this the loop body of the species insertion,
        //	   then I don't have to deal with inserting the nodes into the species node vector.
        bool isCompatible(const Genome& genome, const Species& species)
        {
            //checking for cycles
            //TODO check if this even works xP
            std::vector<TNode> nodes(species.hiddenNodes);

            for (const auto& c : std::views::keys(genome.genes))
            {
                auto f = std::find(nodes.rbegin(), nodes.rend(), c.first);
                auto s = std::find(nodes.rbegin(), nodes.rend(), c.second);

                if (f < s)
                {
                    for (auto it = f + 1; it != s; ++it)
                        if (species.connections.contains(std::pair(*it, *s)))
                            return false;

                    std::rotate(f, f + 1, nodes.rend());
                }
            }

            //compatibility check according to NEAT paper
            std::uint32_t ED =
                std::accumulate(
                    genome.genes.begin(),
                    genome.genes.end(),
                    0,
                    [&species](auto acc, const auto& entry)
                    {
                        return species.genomes[0].genes.contains(entry.first) ? acc : acc + 1;
                    }
                )
                + std::accumulate(
                    species.genomes[0].genes.begin(),
                    species.genomes[0].genes.end(),
                    0,
                    [&genome](auto acc, const auto& entry)
                    {
                        return genome.genes.contains(entry.first) ? acc : acc + 1;
                    }
                );
                    
            std::uint32_t N = std::max(genome.genes.size(), species.genomes[0].genes.size());

            float W = std::accumulate(
                genome.genes.begin(),
                genome.genes.end(),
                0.0f,
                [&species](auto acc, const auto& entry)
                {
                    return
                        species.genomes[0].genes.contains(entry.first)
                        ? acc + std::abs(species.genomes[0].genes[entry.first].weight - entry.second.weight)
                        : acc;
                }
            );

            return c12 * ED / N + c3 * W <= deltaT;
        }

        std::array<float, ToutCount> generateNetworkOutput(
            const NetworkConfiguration& network,
            const std::array<float, TinCount>& input
        )
        {
            std::array<float, ToutCount> finalOutput;
            std::vector<float> output(ToutCount + TinCount + 1 + network.speciesHiddenNodes.size(), 0);
            std::vector<TNode> nodes(network.speciesHiddenNodes);

            std::ranges::copy(input, output.begin() + ToutCount);
            output[ToutCount + TinCount] = 1;

            //TODO don't need this
            auto io = std::views::iota(static_cast<TNode>(1), static_cast<TNode>(TinCount));
            nodes.insert(nodes.end(), io.begin(), io.end());

            //TODO split the loop in 2 so i don't have to copy the vector afterwards
            for (const auto& out : nodes)
                if (network.populationConnections.contains(out))
                    for (const auto& in : network.populationConnections.at(out))
                    {
                        std::pair<TNode, TNode> c(out, in);

                        if (network.genomeGenes.contains(c))
                            if (const auto& s = network.genomeGenes.at(c); s.enabled)
                                output[out] += s.weight * activation(output[in]);
                    }

            std::copy(output.begin(), output.begin() + ToutCount, finalOutput.begin());

            return finalOutput;
        }
    };
}