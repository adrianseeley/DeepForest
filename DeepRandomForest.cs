public class DeepRandomForest
{
    public int yComponent;
    public List<RandomForest> randomForests;

    public DeepRandomForest(List<(List<float> input, List<float> output)> samples, int yComponent, int treeCount, int forestCount, bool extraRandom)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // mark the y component of interest
        this.yComponent = yComponent;

        // initialize the random forests list
        this.randomForests = new List<RandomForest>();

        // create a mutation safe list of training samples
        List<(List<float> input, List<float> output)> trainingSamples = new List<(List<float> input, List<float> output)>();
        foreach ((List<float> input, List<float> output) sample in samples)
        {
            // we deep copy the input and output lists to avoid mutating the originals
            List<float> newInput = new List<float>(sample.input);
            List<float> newOutput = new List<float>(sample.output);
            trainingSamples.Add((newInput, newOutput));
        }

        // iterate through all the forest layers (except the last)
        for (int forestIndex = 0; forestIndex < forestCount - 1; forestIndex++)
        {
            // create a random forest for this layer
            RandomForest randomForest = new RandomForest(trainingSamples, yComponent, treeCount, extraRandom);
            randomForests.Add(randomForest);

            // iterate through all the training samples
            foreach ((List<float> input, List<float> output) trainingSample in trainingSamples)
            {
                // for each tree in the previous forest
                foreach (RandomTree randomTree in randomForest.randomTrees)
                {
                    // add the prediction of the single tree to the input of this training sample (concat)
                    trainingSample.input.Add(randomTree.Predict(trainingSample.input));
                }

                // add the prediction of the ensemble of the previous forest to this training sample (concat)
                trainingSample.input.Add(randomForest.Predict(trainingSample.input));
            }
        }

        // create the last random forest which does not have a subsequent forest
        RandomForest lastRandomForest = new RandomForest(trainingSamples, yComponent, treeCount, extraRandom);
        randomForests.Add(lastRandomForest);
    }

    public float Predict(List<float> input)
    {
        // create a mutation safe list of the input
        List<float> predictInput = new List<float>(input);

        // iterate through all the forest layers (except the last)
        for (int forestIndex = 0; forestIndex < randomForests.Count - 1; forestIndex++)
        {
            // get the forest at the current index
            RandomForest randomForest = randomForests[forestIndex];

            // iterate through all the trees in the forest
            for (int treeIndex = 0; treeIndex < randomForest.randomTrees.Count; treeIndex++)
            {
                // get the tree at the current index
                RandomTree randomTree = randomForest.randomTrees[treeIndex];

                // add the prediction of the single tree to the input (concat)
                predictInput.Add(randomTree.Predict(predictInput));
            }

            // add the prediction of the ensemble to the input (concat)
            predictInput.Add(randomForest.Predict(predictInput));
        }

        // get the last random forest
        RandomForest lastRandomForest = randomForests[randomForests.Count - 1];

        // return the prediction of the last random forest
        return lastRandomForest.Predict(predictInput);
    }
}