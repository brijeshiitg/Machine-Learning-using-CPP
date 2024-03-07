#include <vector>
#include <iostream>
#include "LinearRegression.h"

/** reads input and prints outputs. Interacts with Linear Regression class. */
void inputOutput()
{
    int size;
    double w, b, lr;
    int iterations;
    std::cout << "Enter size of input feature and target: " << std::endl;
    std::cin >> size;
    std::vector<double> x(size), y(size);

    for(unsigned int i = 0; i < size; i++)
    {
        std::cout << "Enter " << i+1 <<"th " << "feature value: ";
        std::cin >> x[i];
    }
    std::cout << std::endl;
    for(unsigned int i = 0; i < size; i++)
    {
        std::cout << "Enter " << i+1 <<"th " << "target value :";
        std::cin >> y[i];
    }

    std::cout << std::endl <<"Enter w: ";
    std::cin >> w;
    std::cout << std::endl <<"Enter b: ";
    std::cin >> b;
    std::cout << std::endl <<"Enter Learning rate: ";
    std::cin >> lr;
    std::cout << std::endl <<"Enter no. of iterations: ";
    std::cin >> iterations;
    std::cout << std::endl;

    LinearRegression linear_regression;
    std::vector<double> final_weigths = linear_regression.gradientDescent(x, y, w, b, lr, iterations);
    std::cout << "W_final = " << final_weigths[0] << "and b_final = "<< final_weigths[1] << std::endl;
}

int main()
{
    inputOutput();
    return 0;
}