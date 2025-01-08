public class ResidualPredictorStandardTree : ResidualPredictor<StandardTree>
{
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public StandardTree.Reduction reduction;

    public ResidualPredictorStandardTree(List<Sample> samples, float learningRate, int minSamplesPerLeaf, int maxLeafDepth, StandardTree.Reduction reduction) 
        : base(samples, learningRate)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.reduction = reduction;
    }

    protected override StandardTree AddModel(List<Sample> residuals)
    {
        StandardTree model = new StandardTree(residuals, minSamplesPerLeaf, maxLeafDepth, reduction);
        return model;
    }

    protected override float[] Predict(StandardTree model, float[] input)
    {
        return model.Predict(input);
    }
}