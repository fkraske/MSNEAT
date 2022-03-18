#pragma once

#include <NEAT/Simulation.h>

namespace MSSolve
{
    class EasyMSSimulation : public Neat::Simulation<std::uint32_t, 900, 100>
    {
    private:
        virtual float activation(float x) override;
        virtual float evaluateFitness(const Network& network, NodeID maxNodeID) override;
        virtual bool shouldFinishTraining(const Snapshot& snapshot) override;

    public:
        EasyMSSimulation();
    };
}