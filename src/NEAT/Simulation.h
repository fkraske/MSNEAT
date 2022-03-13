#pragma once

#include <array>
#include <functional>
#include <numeric>
#include <ranges>

#include "definitions.h"
#include "hashes.h"

//TODO bias (additional input node with constant 1 input)
namespace Neat
{
	template <size_t TinCount, size_t ToutCount, std::unsigned_integral TNode>
	class Simulation
	{
	protected:
		using GenerationCount = std::uint64_t;

		//TODO maybe convert to using definition
		struct Genome
		{
			SpecializationMap<TNode> specializations;
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

			Population(TNode maxNode) :
				maxNode(maxNode),
				connections(),
				species({ Species{ std::vector<Genome>(maxNode, Genome{ }) } })
			{ }
		};

		using FitnessVector = std::vector<std::pair<const Genome&, float>>;

		struct Snapshot
		{
			GenerationCount generation;
			Population& population;
			FitnessVector fitness;
		};

		struct NetworkConfiguration
		{
			const std::vector<TNode>& speciesHiddenNodes;
			const ConnectionMap<TNode>& populationConnections;
			const SpecializationMap<TNode>& genomeSpecializations;
		};

	public:
		//TODO
		struct TrainingResult
		{

		};



	protected:
		std::uint32_t populationSize;

	protected:
		Simulation(std::uint32_t populationSize) : populationSize(populationSize) { }

		virtual std::array<float, TinCount> getNetworkInput() = 0;
		virtual float activation(float f) = 0;
		virtual float evaluateFitness(const NetworkConfiguration& network) = 0;
		virtual bool stopCondition(const Snapshot& snapshot) = 0;

	public:
		TrainingResult train()
		{
			GenerationCount generation = 1;
			auto pp = std::make_unique<Population>(Population{ ToutCount + TinCount + 1 });

			while (true) {
				//Evolution
				Population& p = *pp;

				//Speciation
				//Mutation (Weights, Enabled/Disable, Nodes, Connections)
				//Crossover
				//Culling



				//Fitness calculation
				Snapshot snapshot{ generation, p };

				for (const auto& s : p.species)
					for (const auto& g : s.genomes)
						snapshot.fitness.emplace_back(g, evaluateFitness({ s.hiddenNodes, p.connections, g.specializations }));

				//Loop Condition
				if (stopCondition(snapshot))
					return { }; //TODO

				++generation;
			}

			return { }; //TODO
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

						if (network.genomeSpecializations.contains(c) && network.genomeSpecializations.at(c).enabled)
							output[out] += network.genomeSpecializations.at(c).weight * activation(output[in]);
					}

			std::copy(output.begin(), output.begin() + ToutCount, finalOutput.begin());

			return finalOutput;
		}
	};
}