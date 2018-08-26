#pragma once
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

extern "C" {
xt::xarray<float> run(const xt::xarray<float>&);
xt::xarray<float> predict(const xt::xarray<float>&);
}
