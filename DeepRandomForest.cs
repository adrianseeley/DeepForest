public class DeepRandomForest
{
    public int yComponent;
    public List<RandomForest> randomForests;

    public DeepRandomForest(List<(List<float> input, List<float> output)> samples, int yComponent, int treeCount, int forestCount)
    {
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }
        this.yComponent = yComponent;
        randomForests = new List<RandomForest>();
        List<(List<float> input, List<float> output)> trainingSamples = new List<(List<float> input, List<float> output)>();
        foreach ((List<float> input, List<float> output) sample in samples)
        {
            List<float> newInput = new List<float>(sample.input);
            List<float> newOutput = new List<float>(sample.output);
            trainingSamples.Add((newInput, newOutput));
        }
        for (int forestIndex = 0; forestIndex < forestCount - 1; forestIndex++)
        {
            RandomForest randomForest = new RandomForest(trainingSamples, yComponent, treeCount);
            randomForests.Add(randomForest);
            foreach ((List<float> input, List<float> output) trainingSample in trainingSamples)
            {
                foreach (RandomTree randomTree in randomForest.randomTrees)
                {
                    trainingSample.input.Add(randomTree.Predict(trainingSample.input)); // single tree outputs
                }
                trainingSample.input.Add(randomForest.Predict(trainingSample.input)); // ensemble output
            }
        }
        RandomForest lastRandomForest = new RandomForest(trainingSamples, yComponent, treeCount);
        randomForests.Add(lastRandomForest);
    }

    public float Predict(List<float> input)
    {
        List<float> predictInput = new List<float>(input);
        for (int forestIndex = 0; forestIndex < randomForests.Count - 1; forestIndex++)
        {
            RandomForest randomForest = randomForests[forestIndex];
            for (int treeIndex = 0; treeIndex < randomForest.randomTrees.Count; treeIndex++)
            {
                RandomTree randomTree = randomForest.randomTrees[treeIndex];
                predictInput.Add(randomTree.Predict(predictInput)); // single tree outputs
            }
            predictInput.Add(randomForest.Predict(predictInput)); // ensemble output
        }
        RandomForest lastRandomForest = randomForests[randomForests.Count - 1];
        return lastRandomForest.Predict(predictInput);
    }
}