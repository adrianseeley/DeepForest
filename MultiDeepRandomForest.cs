public class MultiDeepRandomForest
{
    public List<DeepRandomForest> deepRandomForests;

    public MultiDeepRandomForest(List<(List<float> input, List<float> output)> samples, int treeCount, int forestCount)
    {
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }
        deepRandomForests = new List<DeepRandomForest>();
        for (int yIndex = 0; yIndex < samples[0].output.Count; yIndex++)
        {
            DeepRandomForest deepRandomForest = new DeepRandomForest(samples, yIndex, treeCount, forestCount);
            deepRandomForests.Add(deepRandomForest);
        }
    }

    public List<float> Predict(List<float> input)
    {
        List<float> predictOutput = new List<float>();
        foreach (DeepRandomForest deepRandomForest in deepRandomForests)
        {
            predictOutput.Add(deepRandomForest.Predict(input));
        }
        return predictOutput;
    }
}