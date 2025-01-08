public class BaggedPredictorRandomTree : BaggedPredictor<RandomTree>
{
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public int maxSplitAttempts;

    public BaggedPredictorRandomTree(List<Sample> samples, int featuresPerBag, int minSamplesPerLeaf, int maxLeafDepth, int maxSplitAttempts)
        : base(samples, featuresPerBag)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.maxSplitAttempts = maxSplitAttempts;
    }

    protected override RandomTree AddModel(List<Sample> baggedSamples, List<int> baggedFeatures)
    {
        return new RandomTree(baggedSamples, baggedFeatures, minSamplesPerLeaf, maxLeafDepth, maxSplitAttempts);
    }

    protected override float[] Predict(RandomTree model, float[] input)
    {
        return model.Predict(input);
    }
}