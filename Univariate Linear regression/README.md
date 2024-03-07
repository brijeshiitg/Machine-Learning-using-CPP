## 1. Univariate Linear regression:
This directory contains c++ implementation of linar regression with one variable.
### Overview:

**Linear regression model:** $f_{w,b}(x) = w . \vec{x} + b$

**Cost function (average squared error):** $J(w, b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})^2$

**Gradients:**
        $~~~~~~~~~~~~~\frac{\partial{J}}{\partial{w}} = \frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)}).x^{(i)}$
        $~~~~~~~~~~~~~\frac{\partial{J}}{\partial{b}} = \frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})$

**Gradient Descent:**
                
$ ~~~~~~~~~~~repeat \{ $
$~~~~~~~~~~~~~~~~~~~ w = w - \alpha * \frac{\partial{J}}{\partial{w}}$

$~~~~~~~~~~~~~~~~~~~ b = b - \alpha * \frac{\partial{J}}{\partial{b}} $
$~~~~~~~~~~~~~~~~~~~\}simultaneous update $
                
Here, $w$ and $b$ are the learnable parameters, $\alpha$ is the learning rate, $\frac{\partial{J}}{\partial{w}}$ and $\frac{\partial{J}}{\partial{b}}$ are gradients.

### Instructions to run the code:
step 1: go to *Univariate Linear regression* directory and compile and run the code using following command:

        `g++ --std=c++11 *.cpp && ./a.out`
        
step 2: It will prompt you to enter length of the feature and target array. Enter the length according to your requirements.

step 3: Now it will ask you to enter individual elements of feature followed by individual elements of target array.

step 4: Now it will prompt for initial value of w and b.

step 5: It will ask learning rate.

step 6: Finally, it will ask for no. of iterations you want to train the model.

Done!!