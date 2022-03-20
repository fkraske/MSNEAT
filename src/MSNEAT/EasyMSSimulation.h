#pragma once

#include <NEAT/Simulation.h>

namespace MSSolve
{
    // Non-functional skeleton implementation for training networks to solve Minesweeper
    class EasyMSSimulation : public Neat::Simulation<std::uint32_t, 900, 100>
    {
    private:
        virtual float activation(float x) const override;
        virtual float evaluateFitness(const Network& network, NodeID maxNodeID) override;
        virtual bool shouldFinishTraining(Snapshot snapshot) override;
        virtual Population initializePopulation() override;

    public:
        EasyMSSimulation();
    };
}