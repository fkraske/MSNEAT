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
            network.generateOutput({ 0, 0 }, maxNodeID)[0] == 0
            + network.generateOutput({ 1, 0 }, maxNodeID)[0] == 1
            + network.generateOutput({ 0, 1 }, maxNodeID)[0] == 1
            + network.generateOutput({ 1, 1 }, maxNodeID)[0] == 0;
    }

    virtual bool shouldFinishTraining(const Snapshot& snapshot) override
    {
        //if (!(snapshot.generation % 1000)) std::cout << "Generation: " << snapshot.generation << "\n\n" << snapshot.population.basicInfo() << "\n\n" << snapshot.population.networkInfo() << "\n\n\n\n";
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
    XORSimulation s;

    auto r = s.train();

    std::cout << r.population.highestFitness;
}