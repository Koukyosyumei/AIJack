# secure machine learning

## 1. adversarial example
I tested the experiment described in the below paper.

https://arxiv.org/abs/1708.06131

Although the author created adversarial examples against SVM with linear kernel,\
I implemented the attacker against RBF kernel.
The target model is SVM with the RBF kernel which is trained for binary classification
between "3" and "7" of mnist.
The performance of the model is as follows.

                  precision    recall  f1-score   support

               3       0.99      0.99      0.99       229
               7       0.99      0.99      0.99       271

        accuracy                           0.99       500
       macro avg       0.99      0.99      0.99       500
    weighted avg       0.99      0.99      0.99       500


I created an adversarial example which seems "7" for human, but svm can't correctly classify. The result is shown in the following picture.

![](img/output.png)






