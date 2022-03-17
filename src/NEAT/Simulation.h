#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "hashes.h"

//TODO maybe add fitness and generation count types as template parameters
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

		static inline constexpr Fitness defaultFitness = 0;

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

			Fitness combinedAdjustedFitness = defaultFitness;
			Fitness highestFitness = defaultFitness;
			Fitness previousHighestFitness = defaultFitness;

			std::uint32_t remainingOffspring;

			std::uint16_t staleGenerations;
		};

		struct Population
		{
			std::vector<Species> species;

			NodeID maxNodeID;
			ConnectionID maxConnectionID;
			std::unordered_map<ConnectionID, NodeID> nodeInnovations;
			std::unordered_map<Edge, ConnectionID> edgeInnovations;

			Fitness combinedAdjustedFitness = defaultFitness;
			Fitness highestFitness = defaultFitness;
			Fitness previousHighestFitness = defaultFitness;
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

			auto getNetworks()
			{
				return species | std::views::transform([](const auto& s) { return s.networks; }) | std::views::join;
			}

			void insertNetwork(const Species& originSpecies, const Network& network)
			{
				//TODO

				--originSpecies.remainingOffspring;
			}
		};

		struct Network
		{
			std::vector<NodeID> nodes;
			ConnectionMap connections;
			Fitness fitness = defaultFitness;
			Fitness adjustedFitness = defaultFitness;

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

			inline bool isCompatible(const Network& other)
			{
				auto flattened = getEdges();
				auto otherFlattened = other.getEdges();
				auto matching = flattened | std::views::filter([&other](const Edge& e) { return other.containsEdge(e); });

				auto flattenedSize = std::ranges::distance(flattened);
				auto otherFlattenedSize = std::ranges::distance(otherFlattened);
				auto matchingSize = std::ranges::distance(matching);

				auto ED = flattenedSize + otherFlattenedSize - 2 * matchingSize;
				auto N = std::max(flattenedSize, otherFlattenedSize);
				float W = std::accumulate(
					matching.begin(),
					matching.end(),
					0.0f,
					[&other](auto acc, const auto& e)
					{
						return acc + std::abs(other.getConnectionData(e).weight - getConnectionData(e).weight);
					}
				) / matchingSize;

				return c12 * ED / N + c3 * W <= deltaT;
			}

			inline std::array<float, ToutCount> generateOutput(const Population& population, const std::array<float, TinCount>& input)
			{
				std::array<float, ToutCount> finalOutput;
				std::vector<float> output(population.maxNodeID, 0);

				std::ranges::copy(input, output.begin() + ToutCount);
				output[ToutCount + TinCount] = 1;

				for (NodeID out : nodes)
					for (const auto& [in, data] : connections[out])
						output[out] += output[in] * data.weight;

				for (NodeID out = 0; out < ToutCount; ++out)
					for (const auto& [in, data] : connections[out])
						finalOutput[out] += output[in] * data.weight;

				return finalOutput;
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
	private:
		std::uint32_t populationSize;
		std::mt19937_64 rng;
		std::uniform_real_distribution<float> realDistribution;
		float c12;
		float c3;
		float deltaT;
		std::uint16_t stalePopulationLimit;
		std::uint32_t stalePopulationSurvivors;
		std::uint16_t staleSpeciesLimit;
		std::uint32_t championSurvivalLimit;
		float cullFactor;
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
			std::uint32_t championSurvivalLimit,
			float cullFactor,
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
			championSurvivalLimit(championSurvivalLimit),
			cullFactor(cullFactor),
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
			assert(stalePopulationSurvivors > 2);
			assert(staleSpeciesLimit > 0);
			assert(championSurvivalLimit > 0);
			assert(cullFactor >= 0.0f && cullFactor <= 1.0f);
			assert(weightMutateChance > 0.0f);
			assert(perturbChance > 0.0f);
			assert(connectionEnableChance > 0.0f);
			assert(crossoverChance > 0.0f && crossoverChance < 0.0f);
			assert(interspeciesMatingChance > 0.0f);
			assert(addNodeChance > 0.0f);
			assert(addConnectionChance > 0.0f);
		}

		virtual std::array<float, TinCount> getNetworkInput() = 0;
		virtual float activation(float f) = 0;
		virtual Fitness evaluateFitness(const Network& network) = 0; //TODO make member function?
		virtual bool shouldFinishTraining(const Snapshot& snapshot) = 0;

	public:
		TrainingResult train()
		{
			GenerationCount generation = 0;

			Population p;
			p.species.push_back(Species{ std::vector<Network>(populationSize) });
			p.maxNodeID = ToutCount + TinCount + 1;

			do
			{
				// EVOLUTION

				Population pN;
				pN.previousHighestFitness = p.highestFitness;

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

					p.combinedAdjustedFitness = 0;

					for (const Species& s : p.species)
						p.combinedAdjustedFitness += s.combinedAdjustedFitness;
				}
				else {
					// Remove stale Species
					std::ranges::remove_if(p.species, [this](const Species& s) { return s.staleGenerations >= staleSpeciesLimit; });

					if (p.species.empty())
						p.species.push_back(Species{ std::vector<Network>(populationSize) });
				}

				for (Species& s : p.species)
					s.remainingOffspring = std::round(populationSize * s.combinedAdjustedFitness / p.combinedAdjustedFitness);

				// Cull each Species' weakest individuals
				if (cullFactor > 0)
				{
					for (Species& s : p.species)
					{
						std::ranges::sort(s.networks, [](const Network& first, const Network& second) { return second.adjustedFitness - first.adjustedFitness; });
						s.networks.resize(std::clamp<size_t>(s.networks.size() * cullFactor, 1, s.networks.size()));
					}
				}



				// Champions survive
				for (const Species& s : p.species)
					if (s.networks.size() >= championSurvivalLimit)
					{
						pN.insertNetwork(s, s.networks[0]);
					}

				// Interspecies mating
				repeatRandomly(
					[this, &p, &pN]()
					{
						float l1 = getRandomFloat();

						for (auto it1 = p.species.begin(); it1 != p.species.end(); ++it1)
						{
							float rP1 = it1->combinedAdjustedFitness / p.combinedAdjustedFitness;

							if (l1 < rP1)
							{
								float l2 = getRandomFloat() - rP1;

								for (auto it2 = p.species.begin(); it2 != p.species.end(); ++it2 != it1 ? it2 : ++it2)
								{
									float rP2 = it2->combinedAdjustedFitness / (p.combinedAdjustedFitness - it1->combinedAdjustedFitness);

									if (l2 < rP2)
										//TODO incorrect?
									{
										pN.insertNetwork(
											getRandomBool ? *it1 : *it2,
											crossover(
												getRandomPerformanceWeightedParent(*it1),
												getRandomPerformanceWeightedParent(*it2)
											)
										);
									}
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
				{
					while (s.remainingOffspring > 0)
					{
						if (s.networks.size() > 1 && getRandomFloat() > )
						{

						}
					}
				}



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

				p = std::move(pN);



				// FITNESS EVALUATION

				for (auto& s : p.species)
				{
					for (auto& n : s.networks)
					{
						p.highestFitness = std::max(
							p.highestFitness,
							s.highestFitness = std::max(
								s.highestFitness,
								n.fitness = evaluateFitness(n)
							)
						);
						s.combinedAdjustedFitness += n.adjustedFitness = n.fitness / s.networks.size();
					}

					s.staleGenerations = s.highestFitness > s.previousHighestFitness ? 0 : s.staleGenerations + 1;
					p.combinedAdjustedFitness += s.combinedAdjustedFitness;
				}

				p.staleGenerations = p.highestFitness > p.previousHighestFitness ? 0 : p.staleGenerations + 1;
			} while (shouldFinishTraining({ generation++, p }));

			return { p }; //TODO
		}



		//Mutations

		Network mutateWeight(const Network& network)
		{
			Network result(network);
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

			result.getConnectionData(getRandomEdge(result)).enabled = true;

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

				result.getConnectionData(inNode, outNode) = {
					population.getConnectionInnovation(inNode, outNode),
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

			result.getConnectionData(e.first, newNode) = { population.getConnectionInnovation(newNode, e.first), 1.0f, true };
			result.getConnectionData(newNode, e.second) = { population.getConnectionInnovation(e.second, newNode), d.weight, true };
			result.nodes.insert(std::ranges::find(result.nodes, e.second), newNode);

			return result;
		}

		Network crossover(const Network& first, const Network& second)
		{
			Network result;

			const Network& fitter = first.fitness > second.fitness ? first : second;
			const Network& lessFit = first.fitness < second.fitness ? first : second;

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

		inline const Network& getRandomPerformanceWeightedParent(Species& species)
		{
			float l = getRandomFloat();

			for (const Network& n : species.networks)
			{
				float relativePerformance = n.adjustedFitness / species.combinedAdjustedFitness;

				if (l < relativePerformance)
					return n;
				else
					l -= relativePerformance;
			}

			return species.networks.back();
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