#include "EasyMSSimulation.h"

#include <numeric>

float MSSolve::EasyMSSimulation::activation(float x)
{
    //TODO unimplemented
    return 0.0f;
}

float MSSolve::EasyMSSimulation::evaluateFitness(const Network& network, NodeID maxNodeID)
{
    return 0.0f;

    //auto out = generateNetworkOutput(network, getNetworkInput());

    //return std::accumulate(out.begin(), out.end(), 0.0f, [](auto a, auto b) { return a + b; });
}

bool MSSolve::EasyMSSimulation::shouldFinishTraining(const Snapshot& snapshot)
{
    return snapshot.generation > 10000;
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
