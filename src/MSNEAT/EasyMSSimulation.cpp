#include "EasyMSSimulation.h"

#include <numeric>

float MSSolve::EasyMSSimulation::activation(float f)
{
    //TODO unimplemented
    return 0.0f;
}

std::array<float, 900> MSSolve::EasyMSSimulation::getNetworkInput()
{
    return std::array<float, 900>();
}

float MSSolve::EasyMSSimulation::evaluateFitness(const NetworkConfiguration& network)
{
    auto out = generateNetworkOutput(network, getNetworkInput());

    return std::accumulate(out.begin(), out.end(), 0.0f, [](auto a, auto b) { return a + b; });
}

bool MSSolve::EasyMSSimulation::trainingFinishCondition(const Snapshot& snapshot)
{
    return true;
}

MSSolve::EasyMSSimulation::EasyMSSimulation() : Neat::Simulation<900, 100, std::uint32_t>(
    300,
    std::random_device()(),
    0.9f,
    0.1f,
    3.0f,
    1.0f,
    1.0f
) { }
