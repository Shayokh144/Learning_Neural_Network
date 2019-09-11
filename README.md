# Learning_Neural_Network

1. ## [BOOK](http://neuralnetworksanddeeplearning.com/chap1.html)
2. ### [the Gradient](https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/)
3. ### [Partial derivatives/ Gradient khan academy](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/introduction-to-partial-derivatives)
4. ### [3Blue1Brown’s Essence of Linear Algebra](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=1&source=post_page)
5. ### [3Blue1Brown’s Essence of Calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&source=post_page)
6. ### [Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
7. ### [ML-maths_and_others](https://ml-cheatsheet.readthedocs.io/en/latest/calculus.html)
8. ### [the_cross-entropy_cost_function](http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function)
9. ### [overfitting_and_regularization](http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization)
10. ### [Weight Initialization](http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization)

from random import randrange
def createXORTrainTestData(dataRange):
    trainData = []
    testData = []
    countOne = 0
    offset = 5
    i1 = 0
    while i1 <= dataRange:
        i2 = randrange(dataRange)
        if i1 == i2:
            trainData.append(([i1,i2], 0))
        else:
            if countOne > (dataRange/2+ offset):
                i1 = randrange(dataRange)
            else:
                trainData.append(([i1,i2], 1))
                countOne+=1
        if len(trainData) >= dataRange:
            break
        i1+=1
                
    #print(trainData)

    print("label 1 data = ", countOne)
    testData = trainData
    
    return trainData, testData
    
    
    
