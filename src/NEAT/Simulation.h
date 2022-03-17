#pragma once

#include <array>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>
#include <unordered_set>
#include <unordered_map>
#include <vector>

//TODO bias (additional input node with constant 1 input)
namespace Neat
{
	template <std::unsigned_integral TInnovation, TInnovation TinCount, TInnovation ToutCount>
	class Simulation
	{
		// Types
	protected:
		using NodeID = TInnovation;
		using ConnectionID = TInnovation;
		using Edge = std::pair<NodeID, NodeID>;
		using GenerationCount = std::uint64_t;
		using Fitness = float;

		struct ConnectionData
		{
			TInnovation innovation;
			float weight;
			bool enabled;
		};

		using ConnectionMap = std::unordered_map<NodeID, std::unordered_map<NodeID, ConnectionData>>;

		struct Network
		{
			std::vector<NodeID> nodes;
			ConnectionMap connections;
			Fitness fitness;

			inline bool containsEdge(NodeID in, NodeID out) const
			{
				return connections.contains(out) && connections.at(out).contains(in);
			}

			inline bool containsEdge(Edge edge) const
			{
				return containsEdge(edge.first, edge.second);
			}

			inline bool containsEdgeWithInnovation(NodeID in, NodeID out, ConnectionID innovation) const
			{
				return containsEdge(in, out) && connections[out][in].innovation == innovation;
			}

			inline bool containsEdgeWithInnovation(Edge edge, ConnectionID innovation) const
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
					[](const auto& e) {
						return e.second | std::views::transform(
							[&e](const auto& f) {
								return Edge(f.first, e.first);
							});
					})
					| std::views::join;
			}

			const auto getEdges() const
			{
				return connections | std::views::transform(
					[](const auto& e) {
						return e.second | std::views::transform(
							[&e](const auto& f) {
								return Edge(f.first, e.first);
							});
					})
					| std::views::join;
			}

			bool isCompatible(const Network& other)
			{
				auto flattened = getEdges();
				auto otherFlattened = other.getEdges();

				std::uint32_t ED =
					std::accumulate(
						flattened.begin(),
						flattened.end(),
						0,
						[&other](auto acc, const auto& e)
						{
							return acc + other.containsEdge(e);
						}
					)
					+ std::accumulate(
						flattened.begin(),
						flattened.end(),
						0,
						[this](auto acc, const auto& e)
						{
							return acc + containsEdge(e);
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
		};

		struct Species
		{
			std::vector<Network> networks;
			Fitness fitness;
		};

		struct Population
		{
			std::vector<Species> species;

			NodeID maxNodeID;
			ConnectionID maxConnectionID;
			std::unordered_map<ConnectionID, NodeID> nodeInnovations;
			std::unordered_map<Edge, ConnectionID> edgeInnovations;

			Fitness highestFitness; //TODO this shouldn't be in here?
			Fitness previousHighestFitness;
			std::uint16_t staleGenerations;

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
			std::vector<Network> networks;
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
		virtual Fitness evaluateFitness(const Network& network) = 0;
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
			//TODO proper initialization
			auto pp = std::make_unique<Population>(Population{});

			while (true) {
				Population& p = *pp;

				//Fitness calculation
				Snapshot snapshot{ generation, p };

				//for (auto& s : p.species)
				//	for (auto& g : s.genomes)
				//		s.fitness += g.fitness = evaluateFitness({ s.hiddenNodes, p.connections, g.genes }) / s.genomes.size();

				//Loop Condition
				if (shouldFinishTraining(snapshot))
					return { }; //TODO

				++generation;

				//Evolution
				Population pN;

				for (const auto& s : p.species)
				{
					//TODO remove
					for (const auto& n : s.networks)
					{
						mutateWeight(n);
						mutateEnableConnection(n);
						mutateAddConnection(pN, n);
						mutateAddNode(pN, n);
						//crossover(g, g);
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

		Network mutateWeight(const Network& network)
		{
			Network result(network);
			Edge e = getRandomEdge(result);
			ConnectionData& c = result.connections[e.second][e.first];

			c.weight =
				getRandomFloat() < perturbChance
				? c.weight * (1.0f + (getRandomFloat() - 0.5f) * perturbAmount)
				: 2.0f * (getRandomFloat() - 0.5f) * randomWeightSpread;

			return result;
		}

		Network mutateEnableConnection(const Network& network)
		{
			Network result(network);
			Edge e = getRandomEdge(result);

			result.connections[e.second][e.first].enabled = true;

			return result;
		}

		Network mutateAddConnection(Population& population, const Network& network)
		{
			Network result(network);

			size_t inNode = getRandomInt(result.nodes.size() + TinCount + 1);
			size_t outNode = getRandomInt(result.nodes.size() + ToutCount - 1);

			outNode = outNode < result.nodes.size() - 1 ? result.nodes[outNode + (outNode >= inNode)] : outNode - result.nodes.size() + 1;
			inNode = inNode < result.nodes.size() ? result.nodes[inNode] : inNode - result.nodes.size() + ToutCount;

			if (!result.containsEdge(inNode, outNode))
			{
				auto inIt = std::find(result.nodes.rbegin(), result.nodes.rend(), inNode);
				auto outIt = std::find(result.nodes.rbegin(), result.nodes.rend(), outNode);

				if (inIt < outIt)
					if (network.containsIndirectReverseConnection(inIt, outIt))
						std::rotate(inIt, inIt + 1, outIt + 1);
					else
						return result;

				result.connections[outNode][inNode] = {
					population.getConnectionInnovation({ inNode, outNode }),
					(1.0f + (getRandomFloat() - 0.5f)) * initialWeightSpread,
					true
				};
			}

			return result;
		}

		Network mutateAddNode(Population& population, const Network& network)
		{
			Network result(network);

			Edge e = getRandomEdge(result);
			ConnectionData& d = result.connections[e.second][e.first];
			NodeID newNode = population.getNodeInnovation(d.innovation);

			d.enabled = false;

			result.connections[newNode][e.first] = { population.getConnectionInnovation(newNode, e.first), 1.0f, true};
			result.connections[e.second][newNode] = { population.getConnectionInnovation(e.second, newNode), d.weight, true };
			result.nodes.insert(std::ranges::find(result.nodes, e.second), newNode);

			return result;
		}

		Network crossover(const Network& first, const Network& second)
		{
			//TODO innovation
			Network result;

			const Network& fitter = first.fitness > second.fitness ? first : second;
			const Network& lessFit = first.fitness < second.fitness ? first : second;

			result.nodes = fitter.nodes;
			result.connections = fitter.connections;

			for (const auto& p1 : fitter.connections)
				for (const auto& p2 : p1.second)
					if (lessFit.containsEdgeWithInnovation(p2.first, p1.first, fitter.connections.at(p1.first).at(p2.first).innovation) && getRandomInt(2))
						result.connections[p1.first][p2.first] = lessFit.connections[p1.first][p2.first];

			return result;
		}



		std::array<float, ToutCount> generateNetworkOutput(
			const NetworkConfiguration& network,
			const std::array<float, TinCount>& input
		)
		{
			std::array<float, ToutCount> finalOutput;
			std::vector<float> output(ToutCount + TinCount + 1 + network.speciesHiddenNodes.size(), 0);
			std::vector<NodeID> nodes(network.speciesHiddenNodes);

			std::ranges::copy(input, output.begin() + ToutCount);
			output[ToutCount + TinCount] = 1;

			//TODO don't need this
			auto io = std::views::iota(static_cast<NodeID>(1), static_cast<NodeID>(TinCount));
			nodes.insert(nodes.end(), io.begin(), io.end());

			//TODO split the loop in 2 so i don't have to copy the vector afterwards
			for (const auto& out : nodes)
				if (network.populationConnections.contains(out))
					for (const auto& in : network.populationConnections.at(out))
					{
						std::pair<NodeID, NodeID> c(out, in);

						if (network.genomeGenes.contains(c))
							if (const auto& s = network.genomeGenes.at(c); s.enabled)
								output[out] += s.weight * activation(output[in]);
					}

			std::copy(output.begin(), output.begin() + ToutCount, finalOutput.begin());

			return finalOutput;
		}

		Edge getRandomEdge(Network& network)
		{
			auto flattened = network.getEdges();

			return *std::ranges::next(flattened.begin(), getRandomInt(std::ranges::distance(flattened)));
		}
	};
}