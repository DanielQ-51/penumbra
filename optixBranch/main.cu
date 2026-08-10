#include <iostream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include "optixStructs.cuh"
#include "optixSetup.cuh"
#include "hostSetup.cuh"

#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

#ifndef PTX_DIR
#define PTX_DIR "" 
#endif

int main(int argc, char* argv[]) {
    std::cout << "Novum Experimental Optix Branch Launched\n ----------------------------------------------------------------------------------------------\n";

    std::vector<std::string> args(argv + 1, argv + argc);

    if (argc <= 1) {
        std::cout << "Usage: Enter config file path as argument to executable, from project root.\n Example: render.exe configs/config.rendertron \n";
        return 0;
    }

    OptixEngineState engineState;
    initOptixSystem(engineState);

    for (std::string str : args) {
        initRender(engineState,  ASSET_PATH(str), 0);
    }

    optixEngineCleanup(engineState);
    std::cout << "goodbye\n";
    return 0;
}
