835 for preweight
834 for no preweight
this likely indicates its more about the number of sample considered than the combination function
k3 is 9717 with 1/(d+eps)

do a leave one out setup for the validation set, just setup the sample avoidance in the trainer
monte carlo select a input weight or train sample weight to modify, with a random direction for an even and fair distribution?
i think we make a type of weight type, index, direction, triplets, populate the whole thing, shuffle it, and iterate through till the end, shuffle it and repeat until we get no more improvements. gives us a full sweep of all possibilities in a random fair order, without greedy drops, sort of a slow balanced learning rate?
there is also room to add sample weights, not to the distance but to the samples themselves, which could be mixed in in a similar way