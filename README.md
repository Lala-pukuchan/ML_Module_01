# ML_Module_01

## for executing file
```
python3 <filename>
```

## for checking format
```
pycodestyle <filename>
```

## for modifying format
```
black <filename>
```

## for copying docker file
```
docker cp <docker container id>:/app/results .
docker exec -it <CONTAINER_ID> ./test.sh
docker cp <CONTAINER_ID>:/app/results .
```

## tips
- @staticmethod: A decorator in Python that defines a static method in a class. It allows you to write methods inside a class that are not tied to an instance of the class. No self Parameter.

## conclusion
1. What is a hypothesis and what is its goal?

    - A hypothesis in this context is our predicted equation (like a line) based on current data. The goal is to make predictions that are as accurate as possible for new input data.

2. What is the loss function and what does it represent?

    - The loss function measures how accurate our predictions are. It's calculated as the average of the squared differences between the actual data and our predictions. A smaller loss means better accuracy.

3. What is Linear Gradient Descent and what does it do?

    - Gradient Descent is a method used to find the best-fit line for our data. We start with an initial line (determined by theta0 and theta1) and iteratively adjust it. We use the gradient, which is derived from the cost function (often denoted as J), to guide our adjustments. As the gradient gets smaller, our adjustments become finer, helping us to reach the minimum point of the cost function, which is our best-fit line.

4. What happens if you choose a learning rate that is too large?

    - If the learning rate is too large, we might overshoot the minimum of the cost function. Instead of converging to the best solution, our adjustments will be too big and can cause the algorithm to fail in finding the optimal line.

5. What happens if you choose a very small learning rate, but still a sufficient number of cycles?

    - With a very small learning rate, the algorithm will take a longer time to converge to the best solution. It will make very tiny adjustments in each step, so it needs more iterations to reach the minimum of the cost function.

6. Can you explain MSE and what it measures?

    - MSE stands for Mean Squared Error. It's a measure of the average of the squares of the differences (or errors) between our predicted values and the actual values. It helps to quantify how close our hypothesis (predicted line) is to the actual data points. A lower MSE indicates a more accurate fit.
