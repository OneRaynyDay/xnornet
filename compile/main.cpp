#include <xtensor/xnpy.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char** argv){
    if (argc != 2) {
        fprintf(stderr, "Usage: input numpy file, e.x. mnist.npy");
        return EXIT_FAILURE;
    }

    std::string npy_file(argv[1]);

    void *handle = dlopen("./libxnor.so", RTLD_LAZY);    
    if (!handle) {
        fprintf(stderr, "Could not open xnor.so.");
        return EXIT_FAILURE;
    }

    xt::xarray<float> (*run)(const xt::xarray<float>&);
    *(void**)(&run) = dlsym(handle, "run");
    
    xt::xarray<float> (*predict)(const xt::xarray<float>&);
    *(void**)(&predict) = dlsym(handle, "predict");

    xt::xarray<float> x = xt::cast<float>(xt::load_npy<double>(npy_file));
    std::cout << xt::view(predict(x), xt::range(0,10)) << "\n\n";
    return 0;
}
