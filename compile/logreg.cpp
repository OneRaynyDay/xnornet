#include "logreg.h"

xt::xarray<float> run(const xt::xarray<float>& x){
    xt::xarray<float> xnornet_0_linear_weights = xt::cast<float>(xt::load_npy<double>("w.npy"));
    xt::xarray<float> xnornet_0_linear_coefs = xt::cast<float>(xt::load_npy<double>("b.npy"));

    decltype(auto) xnornet_0_linear_dot = xt::linalg::dot(x, xt::transpose(xnornet_0_linear_weights));
    decltype(auto) xnornet_0_linear_add = xnornet_0_linear_dot + xnornet_0_linear_coefs;

    return xt::cast<float>(xnornet_0_linear_add);
}

xt::xarray<float> predict(const xt::xarray<float>& arr){
    decltype(auto) res = run(arr);
    decltype(auto) max_labels = xt::argmax(res, 1);
    return xt::cast<float>(max_labels);
}
