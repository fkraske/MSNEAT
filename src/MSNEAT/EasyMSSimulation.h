#pragma once

#include <NEAT/Simulation.h>

namespace MSSolve
{
    class EasyMSSimulation : public Neat::Simulation<900, 100, std::uint32_t>
    {
    private:
        virtual std::array<float, 900> getNetworkInput() override;
        virtual float activation(float f) override;
        virtual float evaluateFitness(const NetworkConfiguration& network) override;
        virtual bool shouldFinishTraining(const Snapshot& snapshot) override;

    public:
        EasyMSSimulation();
    };
}