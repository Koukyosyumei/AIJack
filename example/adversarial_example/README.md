# Adversarial example

## Evasion Attack

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

![](../../img/output.png)

## Poisoning Attack

Second, I implemented a "poisoning attack" against SVM with a linear kernel.
The data set is the same as section 1, and I referred to the following paper.
You can see that adding only one poisoned image dramatically decreases the accuracy of SVM.

![](../../img/poison_loss.png)
![](../../img/poison_example.png)
