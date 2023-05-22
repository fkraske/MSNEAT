#include <iostream>

#include <MS/Board.h>
#include <MSNEAT/EasyMSSimulation.h>
#include <NEAT/Random.h>

namespace XOR
{


    struct ActivationFunction
    {
        inline float operator()(float x) const
        {
            return 1.0f / (1.0f + std::exp(-4.9f * x));
        }
    };

    using Types = NEAT::Types<std::uint32_t, 2, 1, XOR::ActivationFunction>;

    using NodeID = Types::NodeID;
    using ConnectionID = Types::ConnectionID;
    using Edge = Types::Edge;

    using Population = Types::Population;
    using Species = Types::Species;
    using Network = Types::Network;
    using NodeInnovationManager = Population::NodeInnovationManager;
    using ConnectionInnovationManager = Population::ConnectionInnovationManager;

    struct FitnessFunction
    {
        inline NEAT::Fitness operator()(const Network& network)
        {
            float o00 = network.generateOutput({ 0, 0 })[0];
            float o10 = network.generateOutput({ 1, 0 })[0];
            float o01 = network.generateOutput({ 0, 1 })[0];
            float o11 = network.generateOutput({ 1, 1 })[0];

            NEAT::Fitness f = std::max(0.0f, 4
                - std::abs(o00)
                - std::abs(o10 - 1.0f)
                - std::abs(o01 - 1.0f)
                - std::abs(o11)
            );

            return f * f;
        }
    };

    struct NetworkInitializer
    {
        inline Network operator()(
            NEAT::Random& random,
            NodeInnovationManager& NodeInnovationManager,
            ConnectionInnovationManager& ConnectionInnovationManager
            )
        {
            Network n;

            for (ConnectionID i = 0; i < 3; ++i)
                n.getConnectionData(i + 1, 0) = {
                    ConnectionInnovationManager.getInnovation({ i + 1, 0 }),
                    random.getFloat(),
                    true
                };

            return n;
        }
    };

    struct FinishedEvaluator
    {
        inline bool operator()(const Population::Snapshot& snapshot)
        {
            std::cout << "Generation: " << snapshot.generation << "\n\n" << snapshot.population.basicInfo() << "\n\n\n\n";
            //std::cout << "Generation: " << snapshot.generation << " - Highest Fitness: " << snapshot.population.highestFitness << "\n";

            if (snapshot.generation >= 5000)
                return true;

            Network n = snapshot.population.getFittest();

            return
                std::round(n.generateOutput({ 0, 0 })[0]) == 0.0f &&
                std::round(n.generateOutput({ 1, 0 })[0]) == 1.0f &&
                std::round(n.generateOutput({ 0, 1 })[0]) == 1.0f &&
                std::round(n.generateOutput({ 1, 1 })[0]) == 0.0f;
        }
    };

    using SimulationBase = Types::Simulation<
        XOR::FitnessFunction, XOR::NetworkInitializer, XOR::FinishedEvaluator
    >;

    class Simulation : public SimulationBase
    {
    public:
        Simulation() : SimulationBase (
            150,
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
            0.25f,
            1.0f,
            0.75f,
            0.75f,
            0.75f,
            0.001f,
            0.03f,
            0.05f,
            1.0f
        ) { }
    };


}

int main()
{
    XOR::Simulation sim;
    auto r = sim.train();
    XOR::Network n = r.population.getFittest();

    std::cout << n.detailedInfo() << "\n";
    std::cout << n.generateOutput({ 0, 0 })[0] << "\n";
    std::cout << n.generateOutput({ 1, 0 })[0] << "\n";
    std::cout << n.generateOutput({ 0, 1 })[0] << "\n";
    std::cout << n.generateOutput({ 1, 1 })[0] << "\n";
    
    //XOR::ConnectionInnovationManager m;
    //NEAT::Random r(1);
    //XOR::Network n;

    //for (int i = 0; i < 100; ++i)
    //{
    //    n = n.mutateAddConnection(r, m, 1);
    //}

    //std::cout << n.detailedInfo() << "\n";
}