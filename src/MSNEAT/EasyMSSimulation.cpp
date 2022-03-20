#include "EasyMSSimulation.h"
#include "EasyMSSimulation.h"

#include <numeric>

float MSSolve::EasyMSSimulation::activation(float x) const
{
    //TODO unimplemented
    return 0.0f;
}

float MSSolve::EasyMSSimulation::evaluateFitness(const Network& network, NodeID maxNodeID)
{
    //TODO unimplemented
    return 0.0f;
}

bool MSSolve::EasyMSSimulation::shouldFinishTraining(Snapshot snapshot)
{
    //TODO unimplemented
    return snapshot.generation > 10000;
}

MSSolve::EasyMSSimulation::Population MSSolve::EasyMSSimulation::initializePopulation()
{
    //TODO unimplemented
    return {  };
}

MSSolve::EasyMSSimulation::EasyMSSimulation() : Neat::Simulation<std::uint32_t, 900, 100>(
    300,
    std::random_device()(),
    1.0f,
    0.4f,
    3.0f,
    20,
    2,
    15,
    0.5f,
    5,
    0.5f,
    0.5f,
    3.0f,
    0.9f,
    0.5f,
    1.0f,
    0.75f,
    0.75f,
    0.001f,
    0.08f,
    0.1f,
    1.0f
) { }
