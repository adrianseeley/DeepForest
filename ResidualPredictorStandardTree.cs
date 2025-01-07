public class ResidualPredictorStandardTree : ResidualPredictor<StandardTree>
{
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public StandardTree.Reduction reduction;

    public ResidualPredictorStandardTree(List<Sample> samples, float learningRate, int modelCount, int minSamplesPerLeaf, int maxLeafDepth, StandardTree.Reduction reduction) 
        : base(samples, learningRate, modelCount)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.reduction = reduction;
    }

    protected override (StandardTree model, List<float[]> predictions) AddModel(List<Sample> residuals)
    {
        StandardTree model = new StandardTree(residuals, minSamplesPerLeaf, maxLeafDepth, reduction);
        List<float[]> predictions = new List<float[]>();
        foreach (Sample residual in residuals)
        {
            predictions.Add(model.Predict(residual.input));
        }
        return (model, predictions);
    }

    protected override float[] Predict(StandardTree model, float[] input)
    {
        return model.Predict(input);
    }
}