public class MultiDeepRandomForest
{
    public List<DeepRandomForest> deepRandomForests;

    public MultiDeepRandomForest(List<(List<float> input, List<float> output)> samples, int treeCount, int forestCount, bool extraRandom)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // initialize the deep random forests list
        this.deepRandomForests = new List<DeepRandomForest>();

        // iterate through all the y components
        for (int yIndex = 0; yIndex < samples[0].output.Count; yIndex++)
        {
            // create a deep random forest for this y component
            DeepRandomForest deepRandomForest = new DeepRandomForest(samples, yIndex, treeCount, forestCount, extraRandom);
            deepRandomForests.Add(deepRandomForest);
        }
    }

    public List<float> Predict(List<float> input)
    {
        // initialize the predict output list
        List<float> predictOutput = new List<float>();

        // predict the y component output for each deep random forest
        foreach (DeepRandomForest deepRandomForest in deepRandomForests)
        {
            predictOutput.Add(deepRandomForest.Predict(input));
        }

        // return the predict output list
        return predictOutput;
    }
}