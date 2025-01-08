public class ResidualPredictorRandomTree : ResidualPredictor<RandomTree>
{
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public int maxSplitAttempts;

    public ResidualPredictorRandomTree(List<Sample> samples, float learningRate, int minSamplesPerLeaf, int maxLeafDepth, int maxSplitAttempts) 
        : base(samples, learningRate)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.maxSplitAttempts = maxSplitAttempts;
    }

    protected override RandomTree AddModel(List<Sample> residuals)
    {
        RandomTree model = new RandomTree(residuals, minSamplesPerLeaf, maxLeafDepth, maxSplitAttempts);
        return model;
    }

    protected override float[] Predict(RandomTree model, float[] input)
    {
        return model.Predict(input);
    }
}