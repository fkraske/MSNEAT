#include <iostream>

#include <MS/Board.h>
#include <MSNEAT/EasyMSSimulation.h>

class XORSimulation : public Neat::Simulation<std::uint32_t, 2, 1>
{
    virtual float activation(float x) override
    {
        return 1 / (1 + std::exp(-4.9f * x));
    }

    virtual Fitness evaluateFitness(const Network& network, NodeID maxNodeID) override
    {
        return
            2
            - network.generateOutput({ 0, 0 }, maxNodeID)[0]
            + network.generateOutput({ 1, 0 }, maxNodeID)[0]
            + network.generateOutput({ 0, 1 }, maxNodeID)[0]
            - network.generateOutput({ 1, 1 }, maxNodeID)[0];
    }

    virtual bool shouldFinishTraining(const Snapshot& snapshot) override
    {
        if (snapshot.generation == 300)
            std::cout << "Generation: " << snapshot.generation << "\n\n" << snapshot.population.basicInfo() << "\n\n" << snapshot.population.networkInfo() << "\n\n\n\n";
        else
            std::cout << "Generation: " << snapshot.generation << "\n\n" << snapshot.population.basicInfo() << "\n\n\n\n";

        return snapshot.population.highestFitness > 3.5f;
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
        0.1f,
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

    XORSimulation::Population p{ { { { { }, { } } } } };
    XORSimulation::Species& s = p.species.front();
    XORSimulation::Network& m = s.networks[0];
    XORSimulation::Network& n = s.networks[1];

    sim.repeatRandomly([&sim, &p, &m]() { m = sim.mutateAddConnection(p, m); }, 5, 1);
    sim.repeatRandomly([&sim, &p, &n]() { n = sim.mutateAddConnection(p, n); }, 5, 1);

    std::cout << s.networkInfo() << "\n";

    std::cout << m.generateOutput({ 0, 0 }, p.maxNodeID)[0];
    std::cout << m.generateOutput({ 1, 0 }, p.maxNodeID)[0];
    std::cout << m.generateOutput({ 0, 1 }, p.maxNodeID)[0];
    std::cout << m.generateOutput({ 1, 1 }, p.maxNodeID)[0];


    //std::cout << sim.crossover(m, n).detailedInfo();

    //sim.repeatRandomly([&sim, &p, &m]() { m = sim.mutateAddNode(p, m); }, 1, 1);
    //sim.repeatRandomly([&sim, &p, &n]() { n = sim.mutateAddNode(p, n); }, 1, 1);

    //std::cout << s.networkInfo();

    //std::cout << m.edgeInfo();
    //std::cout << n.edgeInfo();

}