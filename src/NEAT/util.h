#pragma once

namespace NEAT
{


    template <std::floating_point T>
    inline T rescale(T v, T oldMin, T oldMax, T newMin, T newMax)
    {
        if (oldMax == oldMin)
            return v;

        return (v - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;
    }


}