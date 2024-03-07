#include "LinearRegression.h"
#include <iostream>
#include<cmath>

LinearRegression::LinearRegression()
{

}

/** Computes the cost function J(w, b). */
double LinearRegression::costFunction(std::vector<double> input_features, std::vector<double> target, double w, double b)
{
    double cost{0.0f};
    int size = input_features.size();
    for (unsigned int i=0; i<size; i++)
    {
        cost += pow((w * input_features[i] + b - target[i]), 2);
    }
    return cost/(2*size);
}

/** Computes gradients w.r.t. w and b, dJ/dw and dJ/db.*/
std::vector<double> LinearRegression::computeGradients(std::vector<double> input_features, std::vector<double> target, double w, double b)
{
    double dj_dw{0.0}, dj_db{0.0};
    int size = input_features.size();
  
    for (unsigned int i=0; i<size; i++)
    {
        dj_dw += (w * input_features[i] + b - target[i])*input_features[i];
        dj_db += (w * input_features[i] + b - target[i]);
    }
    return std::vector<double> {dj_dw/size, dj_db/size};




}

/** Computes gradient descent. */
std::vector<double> LinearRegression::gradientDescent(std::vector<double> input_features, 
                                    std::vector<double> target, 
                                    double w_initial, 
                                    double b_initial,
                                    float lr,
                                    int iterations)
{
    double w{w_initial}, b{b_initial};
    int size = input_features.size();
    std::vector<double> gradients(2);


    for(unsigned int iteration=0; iteration < iterations; iteration++)
    {
        gradients = computeGradients(input_features, target, w, b);
        w = w - lr * gradients[0];
        b = b - lr * gradients[1];
    
        if (iteration % (iterations / 10) == 0)
        {
            std::cout << "iteration: " << iteration << " | w: " << w << " | b: " << b << " | Cost: " << costFunction(input_features, target, w, b) << std::endl;
        }
    }
    return std::vector<double>{w, b};
}