in both L1 and L2 expressions the effect of regularization is to shrink the weights. 
This accords with our intuition that both kinds of regularization penalize large weights. 
But the way the weights shrink is different. 
In L1 regularization, the weights shrink by a constant amount toward 0. 
In L2 regularization, the weights shrink by an amount which is proportional to w. 
And so when a particular weight has a large magnitude, |w|, 
L1 regularization shrinks the weight much less than L2 regularization does. By contrast, 
when |w| is small, L1 regularization shrinks the weight much more than L2 regularization. 
The net result is that L1 regularization tends to concentrate the weight of the network in 
a relatively small number of high-importance connections, while the other weights are driven toward zero.
