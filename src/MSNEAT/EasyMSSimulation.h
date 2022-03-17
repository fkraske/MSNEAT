#pragma once

#include <NEAT/Simulation.h>

namespace MSSolve
{
    class EasyMSSimulation : public Neat::Simulation<std::uint32_t, 900, 100>
    {
    private:
        virtual std::array<float, 900> getNetworkInput() override;
        virtual float activation(float f) override;
        virtual float evaluateFitness(const Network& network) override;
        virtual bool shouldFinishTraining(const Snapshot& snapshot) override;

    public:
        EasyMSSimulation();
    };
}