public class ResidualPredictorRandomTree : ResidualPredictor<RandomTree>
{
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public int maxSplitAttempts;

    public ResidualPredictorRandomTree(List<Sample> samples, float learningRate, int modelCount, int minSamplesPerLeaf, int maxLeafDepth, int maxSplitAttempts) 
        : base(samples, learningRate, modelCount)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.maxSplitAttempts = maxSplitAttempts;
    }

    protected override (RandomTree model, List<float[]> predictions) AddModel(List<Sample> residuals)
    {
        RandomTree model = new RandomTree(residuals, minSamplesPerLeaf, maxLeafDepth, maxSplitAttempts);
        List<float[]> predictions = new List<float[]>();
        foreach (Sample residual in residuals)
        {
            predictions.Add(model.Predict(residual.input));
        }
        return (model, predictions);
    }

    protected override float[] Predict(RandomTree model, float[] input)
    {
        return model.Predict(input);
    }
}