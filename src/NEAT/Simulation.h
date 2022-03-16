#pragma once

#include <array>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>

#include "definitions.h"
#include "hashes.h"
#include "PerformanceData.h"

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

		//TODO rethink classes
		struct Genome
		{
			SpecializationMap<TNode> genes;
			Fitness fitness; //TODO this shouldn't be in here
		};

		struct Species
		{
			std::vector<Genome> genomes;

			std::vector<TNode> hiddenNodes;
			std::vector<Connection<TNode>> connections;

			Fitness fitness; //TODO this shouldn't be in here
		};

		struct Population
		{
			TNode maxNode;
			ConnectionMap<TNode> connections;

			std::vector<Species> species;

			Fitness lastFitness; //TODO this shouldn't be in here
			std::uint16_t staleGenerations;

			Population(TNode maxNode) : maxNode(maxNode), connections(), species() { }
			Population(const std::vector<Species>& species) : maxNode(ToutCount + TinCount + 1), connections(), species(species) { }
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
		float c12;
		float c3;
		float deltaT;
		std::uint16_t stalePopulationLimit;
		std::uint16_t staleSpeciesLimit;
		std::uint16_t championSurvivalLimit;
		float cullFactor;
		float weightMutateChance;
		float perturbChance;
		float perturbAmount;
		float randomWeightSpread;
		float connectionEnableChance;
		float survivalChance;
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
			std::uint16_t staleSpeciesLimit,
			std::uint16_t championSurvivalLimit,
			float cullFactor,
			float weightMutateChance,
			float perturbChance,
			float perturbAmount,
			float randomWeightSpread,
			float connectionEnableChance,
			float survivalChance,
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
			staleSpeciesLimit(staleSpeciesLimit),
			championSurvivalLimit(championSurvivalLimit),
			cullFactor(cullFactor),
			weightMutateChance(weightMutateChance),
			perturbChance(perturbChance),
			perturbAmount(perturbAmount),
			randomWeightSpread(randomWeightSpread),
			connectionEnableChance(connectionEnableChance),
			survivalChance(survivalChance),
			interspeciesMatingChance(interspeciesMatingChance),
			addNodeChance(addNodeChance),
			addConnectionChance(addConnectionChance),
			initialWeightSpread(initialWeightSpread)
		{ }

		virtual std::array<float, TinCount> getNetworkInput() = 0;
		virtual float activation(float f) = 0;
		virtual Fitness evaluateFitness(const NetworkConfiguration& network) = 0;
		virtual bool shouldFinishTraining(const Snapshot& snapshot) = 0;

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
			auto pp = std::make_unique<Population>(Population(std::vector{ Species{ std::vector{ Genome{ } } } }));

			while (true) {
				Population& p = *pp;

				//Fitness calculation
				Snapshot snapshot{ generation, p };

				for (auto& s : p.species)
					for (auto& g : s.genomes)
						s.fitness += g.fitness = evaluateFitness({ s.hiddenNodes, p.connections, g.genes }) / s.genomes.size();

				//Loop Condition
				if (shouldFinishTraining(snapshot))
					return { }; //TODO

				++generation;

				//Evolution
				Population pN(p.maxNode);

				for (const auto& s : p.species)
				{
					//TODO remove
					for (const auto& g : s.genomes)
					{
						mutateWeight(g);
						mutateEnableConnection(g);
						mutateAddConnection(g, s, pN.maxNode);
						mutateAddNode(g, pN.maxNode);
						crossover(g, g);
					}
					//TODO stale population
					//TODO stale species
					//TODO culling
					//TODO champion survival
					//TODO crossover
					//TODO mutation
				}

				pp = std::make_unique<Population>(pN);
			}

			return { }; //TODO
		}



		//Mutations

		Genome mutateWeight(const Genome& genome)
		{
			Genome result(genome);

			auto it = std::next(result.genes.begin(), getRandomInt(result.genes.size()));

			it->second.weight =
				getRandomFloat() < perturbChance
				? it->second.weight * (1.0f + (getRandomFloat() - 0.5f) * perturbAmount)
				: 2.0f * (getRandomFloat() - 0.5f) * randomWeightSpread;

			return result;
		}

		Genome mutateEnableConnection(const Genome& genome)
		{
			Genome result(genome);

			auto v = std::views::filter(result.genes, [](const auto& g) { return !g.second.enabled; });
			std::next(v.begin(), getRandomInt(std::distance(v.begin(), v.end())))->second.enabled = true;

			return result;
		}

		Genome mutateAddConnection(const Genome& genome, const Species& species, TNode maxNode)
		{
			//TODO add the connection to population and species
			Genome result(genome);

			size_t inNode = getRandomInt(maxNode - ToutCount) + ToutCount;
			size_t outNode = getRandomInt(maxNode - TinCount - 2);

			if (outNode >= ToutCount)
			{
				outNode += TinCount + 1;
				outNode += (outNode >= inNode);
			}

			Connection<TNode> c{
				static_cast<TNode>(inNode),
				static_cast<TNode>(outNode)
			};

			if (!result.genes.contains(c))
			{
				auto f = std::find(species.hiddenNodes.rbegin(), species.hiddenNodes.rend(), c.first);
				auto s = std::find(species.hiddenNodes.rbegin(), species.hiddenNodes.rend(), c.second);

				if (f < s)
					for (auto it = f + 1; it != s; ++it)
						if (result.genes.contains(std::pair(*it, *s)))
							return result;

				result.genes[c] = { (1.0f + (getRandomFloat() - 0.5f)) * initialWeightSpread, true };
			}

			return result;
		}

		//NOTE this manipulates maxNode
		Genome mutateAddNode(const Genome& genome, TNode& maxNode)
		{
			//TODO add the node into the gene node order
			Genome result(genome);

			auto it = std::next(result.genes.begin(), getRandomInt(result.genes.size()));

			it->second.enabled = false;

			result.genes[std::pair<TNode, TNode>(it->first.first, maxNode)] = { 1.0f, true };
			result.genes[std::pair<TNode, TNode>(maxNode, it->first.second)] = { it->second.weight, true };

			++maxNode;

			return result;
		}

		Genome crossover(const Genome& first, const Genome& second)
		{
			Genome result;

			const Genome& fitter = first.fitness > second.fitness ? first : second;
			const Genome& lessFit = first.fitness < second.fitness ? first : second;

			for (const auto& g : fitter.genes)
				if (lessFit.genes.contains(g.first))
					result.genes[g.first] = getRandomInt(2) ? g.second : lessFit.genes.at(g.first);
				else
					result.genes[g.first] = g.second;

			return result;
		}



		//TODO maybe swap cycle and Neat-based compatibility check and just make this the loop body of the species insertion,
		//	   then I don't have to deal with inserting the nodes into the species node vector again, since i already do it in the neat compatibility check
		bool isCompatible(const Genome& genome, const Species& species)
		{
			//checking for cycles
			//TODO check if this even works xP
			std::vector<TNode> nodes(species.hiddenNodes);
			std::vector<Connection<TNode>> connections(species.connections);

			for (const auto& c : std::views::keys(genome.genes))
			{
				auto f = std::find(nodes.rbegin(), nodes.rend(), c.first);
				auto s = std::find(nodes.rbegin(), nodes.rend(), c.second);

				if (f < s)
				{
					for (auto it = f + 1; it != s; ++it)
						if (connections.contains(std::pair(*it, *s)))
							return false;

					std::rotate(f, f + 1, nodes.rend());
				}

				connections.push_back(c);
			}

			//compatibility check according to NEAT paper
			std::uint32_t ED =
				std::accumulate(
					genome.genes.begin(),
					genome.genes.end(),
					0,
					[&species](auto acc, const auto& entry)
					{
						return acc + species.genomes[0].genes.contains(entry.first);
					}
				)
				+ std::accumulate(
					species.genomes[0].genes.begin(),
					species.genomes[0].genes.end(),
					0,
					[&genome](auto acc, const auto& entry)
					{
						return acc + genome.genes.contains(entry.first);
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