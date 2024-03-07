#pragma once
#include<vector>

class LinearRegression
{
    public:
    LinearRegression();

    /** Computes linear function f(x) = w . x + b*/
    static std::vector<double> modelFunction(std::vector<double> input_features,
                        double w, 
                        double b);
    
    /** Computes the cost function J(w, b). */
    static double costFunction(std::vector<double> input_features, 
                        std::vector<double> target, 
                        double w, 
                        double b);
    /** Computes gradients w.r.t. w and b, dJ/dw and dJ/db.*/
    static std::vector<double> computeGradients(std::vector<double> input_features, 
                                         std::vector<double> target, 
                                         double w, 
                                         double b);
    /** Computes gradient descent. */
    static std::vector<double> gradientDescent(std::vector<double> input_features, 
                                    std::vector<double> target, 
                                    double w_initial, 
                                    double b_initial,
                                    float lr,
                                    int iterations);

};