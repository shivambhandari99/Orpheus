# Orpheus

1\. MSE and SSIM values for all models in the table below .

<img width="460" alt="Screenshot 2022-07-13 at 11 29 50 AM" src="https://user-images.githubusercontent.com/35135771/178784541-132c973d-eebc-433e-8ef6-6cfa91f176c9.png">

2\. Two plots of training loss in the TensorBoard, where one plot is for moving MNIST models and one plot is for SST models. 
One plot is showing two curves (i.e., w/wo the teacher forcing
strategy), like the figure below.

**Moving MNIST**

![image](https://user-images.githubusercontent.com/35135771/178784638-5f07f876-3e01-49b2-ac12-013e5778a7a0.png)

-   **with teacher forcing (Orange)**

-   **without teacher forcing (Yellow)**

**SST**

![image](https://user-images.githubusercontent.com/35135771/178784657-9aab5ef9-71c6-45d2-b461-a6c8f3b55984.png)

-   **without teacher forcing (Green)**

-   **with teacher forcing (Orange)**

3\. Findings and discussion on model performance with two datasets
and how teacher forcing affects convergence speed and accuracy 

My teacher forcing technique was based on observing the convergence for
the regular method and then when there's an acceptable level of accuracy
observed, using the number of epochs that got me there as a reference to
update for my teacher forcing. For example, my moving_mnist data
converged at around 100 epochs without teacher forcing, so I set up my
teacher forcing probability to change from 1 to 0 linearly as we moved
from 1st epoch to the 100th epoch, updating each epoch.

The loss starts out really low because initially the model is being fed
a lot of ground truth which makes the outputs closer to real, whereas in
case of normal learning, the predictions are okay for the first output
frame, but get progressively worse, which makes the loss much more
significant. But, as the model is fed less ground truth as epochs elapse
the loss increases from its previous levels.

Based on my results, the teacher forcing method gives better performance
on the MSE metric for the same number of epochs across both the
datasets. This shows that teacher forcing converges faster. Both the
methods show similar levels of SSIM across both the datasets. Therefore
teacher forcing is objectively better in the experiment conditions of
this assignment.
