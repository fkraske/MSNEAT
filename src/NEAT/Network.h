#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "constants.h"
#include "types.h"
#include "Random.h"

namespace NEAT
{


    template <
        std::unsigned_integral TInnovation, TInnovation TinCount, TInnovation ToutCount,
        std::default_initializable TActivationFunction
    >
    class Network
    {
    public:
        using NodeID = InnovationTypes<TInnovation>::NodeID;
        using ConnectionID = InnovationTypes<TInnovation>::ConnectionID;
        using Edge = InnovationTypes<TInnovation>::Edge;
        using ConnectionData = InnovationTypes<TInnovation>::ConnectionData;
        using NodeInnovationManager = InnovationTypes<TInnovation>::NodeInnovationManager;
        using ConnectionInnovationManager = InnovationTypes<TInnovation>::ConnectionInnovationManager;

        using ConnectionMap = std::unordered_map<NodeID, std::unordered_map<NodeID, ConnectionData>>;

        Network() :
            fitness(MINIMUM_FITNESS),
            adjustedFitness(MINIMUM_FITNESS),
            weight(0),
            nodes(),
            maxNodeID(ToutCount + TinCount + 1),
            connections(),
            activation()
        { }
        Network(Network& other) = default;
        Network(const Network& other) = default;

        Network& operator=(const Network& other)
        {
            fitness = other.fitness;
            adjustedFitness = other.adjustedFitness;
            weight = other.weight;
            nodes = other.nodes;
            maxNodeID = other.maxNodeID;
            connections = other.connections;
            activation = other.activation;

            return *this;
        } // TODO should just be = default

        std::array<float, ToutCount> generateOutput(const std::array<float, TinCount>& input) const
        {
            std::array<float, ToutCount> finalOutput{ };
            std::vector<float> output(maxNodeID, 0);

            std::ranges::copy(input, output.begin() + ToutCount);
            output[ToutCount + TinCount] = 1;

            for (NodeID hidden : nodes)
            {
                bool applyActivation = false;

                for (const auto& [in, data] : connections.at(hidden))
                    if (data.enabled)
                    {
                        output[hidden] += output[in] * data.weight;
                        applyActivation = true;
                    }

                if (applyActivation)
                    output[hidden] = activation(output[hidden]);
            }

            for (NodeID out = 0; out < ToutCount; ++out)
            {
                if (connections.contains(out))
                    for (const auto& [in, data] : connections.at(out))
                        if (data.enabled)
                            finalOutput[out] += output[in] * data.weight; // TODO separate activation function for output layer
                        }

            return finalOutput;
        }

        bool isCompatible(const Network& other, float c12, float c3, float deltaT) const
        {
            auto flattened = getEdges();
            auto otherFlattened = other.getEdges();
            auto matching = flattened | std::views::filter([&other](const Edge& e) { return other.containsEdge(e); });

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
                W += std::abs(other.getConnectionData(e).weight - getConnectionData(e).weight);

            W /= matchingSize;

            return c12 * ED / N + c3 * W <= deltaT;
        }

        Network mutateWeight(
            Random& random,
            float perturbChance,
            float perturbAmount,
            float randomWeightSpread
        ) const
        {
            Network result(*this);

            auto edges = result.getEdges();

            if (edges.begin() == edges.end())
                return result;

            ConnectionData& c = result.getConnectionData(result.getRandomEdge(random));

            c.weight =
                random.getFloat() < perturbChance
                ? c.weight * (1.0f + (random.getFloat() - 0.5f) * perturbAmount)
                : 2.0f * (random.getFloat() - 0.5f) * randomWeightSpread;

            return result;
        }

        Network mutateEnableConnection(Random& random) const
        {
            Network result(*this);

            auto disabledEdges = result.getEdges() | std::views::filter([&result](Edge e) { return !result.getConnectionData(e).enabled; });
            auto disabledSize = std::ranges::distance(disabledEdges);

            if (disabledSize > 0)
                result.getConnectionData(*std::ranges::next(disabledEdges.begin(), random.getUInt64(disabledSize))).enabled = true;

            return result;
        }

        Network mutateAddConnection(
            Random& random,
            ConnectionInnovationManager& connectionInnovationManager,
            float initialWeightSpread
        ) const
        {
            Network result(*this);

            size_t nodesSize = result.nodes.size();

            size_t inNode = random.getUInt64(nodesSize + TinCount + 1);
            size_t outNode = random.getUInt64(nodesSize + ToutCount - (inNode < nodesSize));

            outNode += inNode < nodesSize && inNode <= outNode;

            if (inNode >= nodesSize || outNode >= nodesSize)
            {
                inNode = inNode < nodesSize ? result.nodes[inNode] : inNode + ToutCount - nodesSize;
                outNode = outNode < nodesSize ? result.nodes[outNode] : outNode - nodesSize;
            }
            else
            {
                auto inIt = result.nodes.rbegin() + result.nodes.size() - 1 - inNode;
                auto outIt = result.nodes.rbegin() + result.nodes.size() - 1 - outNode;

                inNode = result.nodes[inNode];
                outNode = result.nodes[outNode];

                if (inIt < outIt)
                    if (containsIndirectReverseConnection(inIt, outIt))
                        std::rotate(inIt, inIt + 1, outIt + 1);
                    else
                        return result;
            }

            if (!result.containsEdge(inNode, outNode))
                result.getConnectionData(inNode, outNode) = {
                    connectionInnovationManager.getInnovation({ inNode, outNode }),
                    (1.0f + (random.getFloat() - 0.5f)) * initialWeightSpread,
                    true
                };

            return result;
        }

        Network mutateAddNode(
            Random& random,
            NodeInnovationManager& nodeInnovationManager,
            ConnectionInnovationManager& connectionInnovationManager
        ) const
        {
            Network result(*this);

            auto edges = result.getEdges();

            if (edges.begin() == edges.end())
                return result;

            Edge e = result.getRandomEdge(random);
            ConnectionData& d = result.connections[e.second][e.first];
            NodeID newNode = nodeInnovationManager.getInnovation(d.innovation);

            d.enabled = false;

            result.getConnectionData(e.first, newNode) = { connectionInnovationManager.getInnovation({ newNode, e.first }), 1.0f, true };
            result.getConnectionData(newNode, e.second) = { connectionInnovationManager.getInnovation({ e.second, newNode }), d.weight, true };
            result.nodes.insert(std::ranges::find(result.nodes, e.second), newNode);
            result.maxNodeID = std::max(maxNodeID, newNode + 1);

            return result;
        }

        Network crossover(const Network& other, Random& random) const
        {
            Network result;

            const Network& moreFit = fitness > other.fitness ? *this : other;
            const Network& lessFit = fitness <= other.fitness ? *this : other;

            result.nodes = moreFit.nodes;
            result.maxNodeID = moreFit.maxNodeID;
            result.connections = moreFit.connections;

            for (Edge e : moreFit.getEdges())
                if (lessFit.containsEdgeWithInnovation(e, moreFit.getConnectionData(e).innovation) && random.getBool())
                    result.getConnectionData(e) = lessFit.getConnectionData(e);

            return result;
        }

        inline Edge getRandomEdge(Random& random)
        {
            auto flattened = getEdges();
            return *std::ranges::next(flattened.begin(), random.getUInt64(std::ranges::distance(flattened)));
        }

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
                            const auto& in = p2.first;
                            return Edge(in, out);
                        }
                    );
                }) | std::views::join;
        }

        auto getEdges() const
        {
            return connections | std::views::transform(
                [](const auto& p1) {
                    const auto& [out, connected] = p1;

                    return connected | std::views::transform(
                        [&out](const auto& p2) {
                            const auto& in = p2.first;
                            return Edge(in, out);
                        }
                    );
                }) | std::views::join;
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

        std::string connectionInfo(int indent = 0) const
        {
            std::stringstream ss;

            ss << std::string(indent, '\t') << "{\n";

            for (const auto e : getEdges())
            {
                const ConnectionData& data = getConnectionData(e);

                ss  << std::string(indent + 1, '\t') << "{ "
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

        Fitness fitness;
        Fitness adjustedFitness;
        float weight;

    private:
        std::vector<NodeID> nodes;
        NodeID maxNodeID; // TODO should be supNodeID
        ConnectionMap connections;
        TActivationFunction activation;
    };


}