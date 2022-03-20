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
        std::cout << "Generation: " << snapshot.generation << "\n\n" << snapshot.population.basicInfo() << "\n\n\n\n";
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
}