#pragma once

#include "InnovationManager.h"

namespace NEAT
{


    template <typename TInnovation>
    struct InnovationTypes // TODO rename
    {
        using NodeID = TInnovation;
        using ConnectionID = TInnovation;
        using Edge = std::pair<NodeID, NodeID>;

        struct ConnectionData
        {
            ConnectionID innovation;
            float weight;
            bool enabled;
        };

        using NodeInnovationManager = InnovationManager<ConnectionID, NodeID>;
        using ConnectionInnovationManager = InnovationManager<Edge, ConnectionID>;
    };
    
    using GenerationCount = std::uint64_t;
    using Fitness = float;


}