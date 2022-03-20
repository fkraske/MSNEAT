#include <iostream>

#include <MS/Board.h>
#include <MSNEAT/EasyMSSimulation.h>

class XORSimulation : public Neat::Simulation<std::uint32_t, 2, 1>
{
    virtual float activation(float x) const override
    {
        return 1 / (1 + std::exp(-4.9f * x));
    }

    virtual Fitness evaluateFitness(const Network& network, NodeID maxNodeID) override
    {
        Fitness f = 4
            - std::abs(generateOutput(network, { 0, 0 }, maxNodeID)[0])
            - std::abs(generateOutput(network, { 1, 0 }, maxNodeID)[0] - 1)
            - std::abs(generateOutput(network, { 0, 1 }, maxNodeID)[0] - 1)
            - std::abs(generateOutput(network, { 1, 1 }, maxNodeID)[0]);

        return f * f;
    }

    virtual bool shouldFinishTraining(Snapshot snapshot) override
    {
        //if (!(snapshot.generation % 100))
        //    std::cout << "Generation: " << snapshot.generation << "\n\n" << snapshot.population.basicInfo() << "\n\n" << snapshot.population.networkInfo() << "\n\n\n\n";
        //else
        std::cout << "Generation: " << snapshot.generation << "\n\n" << snapshot.population.basicInfo() << "\n\n\n\n";

        //size_t maxNodes = 0;
        //size_t maxEdges = 0;

        //for (Species& s : snapshot.population.species)
        //    for (Network& n : s.networks)
        //    {
        //        maxNodes = std::max<size_t>(maxNodes, n.nodes.size());
        //        maxEdges = std::max<size_t>(maxEdges, std::ranges::distance(n.getEdges()));
        //    }

        //std::cout << maxNodes << "\n";
        //std::cout << maxEdges << "\n";

        return snapshot.population.highestFitness > 15.5f;
    }

    virtual Population initializePopulation() override
    {
        Population p;

        for (int i = 0; i < populationSize; ++i)
        {
            Network n;

            for (ConnectionID i = 0; i < 3; ++i)
                n.getConnectionData(i + 1, 0) = {
                    p.getConnectionInnovation(i + 1, 0),
                    getRandomFloat() * initialWeightSpread,
                    true
                };

            Species s;

            insertNetwork(p, s, n);
        }

        p.nodeInnovations.clear();
        p.edgeInnovations.clear();

        return p;
    }

public:
    XORSimulation() : Simulation(
        300,
        std::random_device()(),
        1.0f,
        0.4f,
        3.0f,
        20,
        2,
        15,
        0.2f,
        5,
        0.5f,
        0.2f,
        0.8f,
        0.9f,
        0.5f,
        1.0f,
        0.75f,
        0.75f,
        0.001f,
        0.03f,
        0.05f,
        1.0f
    ) { }
};

int main()
{
    XORSimulation sim;
    auto r = sim.train();
    XORSimulation::Network n = r.population.getFittest();

    std::cout << n.detailedInfo();
    /*std::cout << std::abs(n.generateOutput({ 0, 0 }, r.population.maxNodeID)[0]);
    std::cout << std::abs(1 - n.generateOutput({ 1, 0 }, r.population.maxNodeID)[0]);
    std::cout << std::abs(1 - n.generateOutput({ 0, 1 }, r.population.maxNodeID)[0]);
    std::cout << std::abs(n.generateOutput({ 1, 1 }, r.population.maxNodeID)[0]);*/

    //XORSimulation::Population p{ { { { { }, { } } } }, 4 };
    //XORSimulation::Species& s = p.species.front();
    //XORSimulation::Network& m = s.networks[0];
    //XORSimulation::Network& n = s.networks[1];

    //sim.repeatRandomly([&sim, &p, &m]() { m = sim.mutateAddConnection(p, m); }, 3, 1);
    //sim.repeatRandomly([&sim, &p, &n]() { n = sim.mutateAddConnection(p, n); }, 3, 1);

    //sim.repeatRandomly([&sim, &p, &m]() { m = sim.mutateAddNode(p, m); }, 2, 1);
    //sim.repeatRandomly([&sim, &p, &n]() { n = sim.mutateAddNode(p, n); }, 2, 1);

    //sim.repeatRandomly([&sim, &p, &m]() { m = sim.mutateAddConnection(p, m); }, 5, 1);
    //sim.repeatRandomly([&sim, &p, &n]() { n = sim.mutateAddConnection(p, n); }, 5, 1);

    //sim.repeatRandomly([&sim, &p, &m]() { m = sim.mutateWeight(m); }, 5, 1);
    //sim.repeatRandomly([&sim, &p, &n]() { n = sim.mutateWeight(n); }, 5, 1);

    //std::cout << s.networkInfo() << "\n";

    //std::cout << m.generateOutput({ 0, 0 }, p.maxNodeID)[0] << std::endl;
    //std::cout << m.generateOutput({ 1, 0 }, p.maxNodeID)[0] << std::endl;
    //std::cout << m.generateOutput({ 0, 1 }, p.maxNodeID)[0] << std::endl;
    //std::cout << m.generateOutput({ 1, 1 }, p.maxNodeID)[0] << std::endl;


    //std::cout << sim.crossover(m, n).detailedInfo();


    //std::cout << s.networkInfo();

    //std::cout << m.edgeInfo();
    //std::cout << n.edgeInfo();

}