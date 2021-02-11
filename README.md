# secure machine learning

## 1. membership inference

I implemented the membership inference for pytorch and scikit-learn.

        example
                sm = ShadowModel([Net(), Net(),Net(), Net(), Net()], 400, shadow_transform=transform)
                result = sm.fit_transform(X_test, y_test, num_itr=10)

                models = [SVC() for i in range(len(result.keys()))]
                am = AttackerModel(models)
                am.fit(result)

My implementation is mainly based on this paper.

https://arxiv.org/abs/1610.05820


## 2. model inversion

The following paper suggest the methods to exract the training data from the output of the model.

https://dl.acm.org/doi/pdf/10.1145/2810103.2813677

I implemented this method for pytorch model and test it on AT&T Database of Faces described in the paper.
The usage of my implementation is as follows.

        example
                mi = Model_inversion(torch_model, input_shape)
                x_result, log = mi.attack(target_label, step_size, number_of_iterations)

You can see some results on AT&T dataset.

![](img/model_inversion.png)


## 3. Evasion Attack
I tested the experiment described in the below paper.

https://arxiv.org/abs/1708.06131

Although the author created adversarial examples against SVM with linear kernel,\
I also implemented the attacker against RBF and poly kernel.
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

The usage of my code is really simple.

        example

            # datasets which contains only "3"
            X_minus_1 = X_train[np.where(y_train == "3")]

            # Attack_sklearn automatically detect the type of the classifier
            attacker = Attack_sklearn(clf = clf, X_minus_1 = X_minus_1,
                                      dmax =  (5000 / 255) * 2.5,
                                      max_iter = 300,
                                      gamma = 1 / (X_train.shape[1] * np.var(X_train)),
                                      lam = 10, t = 0.5, h = 10)

            # x0 is the intial ponint ("7")
            xm, log = attacker.attack(x0)


## 4. Poisoning Attack

Second, I implemented a "poisoning attack" against SVM with a linear kernel.
The data set is the same as section 1, and I referred to the following paper.
https://arxiv.org/abs/1206.6389

You can see that adding only one poisoned image dramatically decreases the accuracy of SVM.

![](img/poison_loss.png)
![](img/poison_example.png)

The usage of my code is as follows.

        example

            attacker = Poison_attack_sklearn(clf,
                                            X_train_, y_train_,
                                            t=0.5)

            xc_attacked, log = attacker.attack(xc, 1,
                                            X_valid, y_valid_,
                                            num_iterations=200)








