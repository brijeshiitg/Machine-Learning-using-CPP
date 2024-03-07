#include <vector>
#include <iostream>
#include "LinearRegression.h"

/** reads input and prints outputs. Interacts with Linear Regression class. */
void inputOutput()
{
    int size;
    double w, b, lr;
    unsigned int iterations;
    char userOption;
    
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

    std::cout << "Do you want to predict the output ? Press Y for yes, any other key for no :";
    std::cin >> userOption;
    
    if (userOption == 'Y' || userOption == 'y')
    {
        unsigned int test_size;
        std::cout << std::endl << "Enter size of input vector you want to predict: ";
        std::cin >> test_size;
        std::vector<double> test_data(test_size), output_data;

        for (unsigned int i = 0; i < test_size; i++)
        {
            std::cout << std::endl << "Enter " << i+1 << "th element: ";
            std::cin >> test_data[i];
        }
        output_data = LinearRegression::modelFunction(test_data, final_weigths[0], final_weigths[1]);

        std::cout << std::endl << " Your prediction is: ";
        for (unsigned int i = 0; i < test_size; i++)
        {
            std::cout << output_data[i] << "  ";
        }
        std::cout << std::endl;

    }
}

int main()
{
    inputOutput();
    return 0;
}