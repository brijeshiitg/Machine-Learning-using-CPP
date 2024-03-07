#include <vector>
#include <iostream>
#include <cmath>


/** Computes the linear function f(x) = w.x + b */
std::vector<double> linearFunction(std::vector<double> input_features, double w, double b)
{
    int size = input_features.size();
    std::vector<double> f_wb(size);

    for (unsigned int i=0; i<size; i++)
    {
        f_wb[i] = w * input_features[i] + b;
    }
    return f_wb;
}

/** Computes the cost function J(w, b). */
double costFunction(std::vector<double> input_features, std::vector<double> target, double w, double b)
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
std::vector<double> computeGradients(std::vector<double> input_features, std::vector<double> target, double w, double b)
{
    double dj_dw{0.0}, dj_db{0.0};
    int size = input_features.size();
    for (unsigned int i=0; i<size; i++)
    {
        dj_dw += (w * input_features[i] + b - target[i])*input_features[i];
        dj_dw += (w * input_features[i] + b - target[i]);
    }
    return std::vector<double> {dj_dw/size, dj_db/size};




}

/** Computes gradient descent. */
std::vector<double> gradientDescent(std::vector<double> input_features, 
                                    std::vector<double> target, 
                                    double w_initial, 
                                    double b_initial,
                                    float lr,
                                    int iterations)
{
    double w{w_initial}, b{b_initial};
    int size = input_features.size();
    std::vector<double> gradients;


    for(unsigned int iteration=0; iteration < iterations; iteration++)
    {
        gradients = computeGradients(input_features, target, w, b);
        w = w - lr * gradients[0];
        b = b - lr * gradients[1];
        if (iteration % (iterations / 10) == 0)
        {
            std::cout << "iteration: " << iteration << "| w: " << w << "| b: " << b << "| Cost: " << costFunction(input_features, target, w, b) << std::endl;
        }
    }
    return std::vector<double>{w, b};
}


int main()
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
        std::cin >> x[i];
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
    // calling linearFunction
    // std::vector<double> f_wb = linearFunction(x, w, b);
    // std::cout << "f_wb(x) = ";
    // for (auto& i : f_wb)
    // {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;
    // // calling costFunction
    // std::cout << "Cost: " << costFunction(x, y, w, b) << std::endl;
    // // calling computeGradients:
    // std::cout << "dJ/dw = " << computeGradients(x, y, w, b)[0] << " dJ/db = " << computeGradients(x, y, w, b)[1] << std::endl;
    // calling gradientDescent:
    std::vector<double> final_weigths = gradientDescent(x, y, w, b, lr, iterations);
    std::cout << "final w and b are: " << final_weigths[0] << " " << final_weigths[1] << std::endl;
    return 0;
}