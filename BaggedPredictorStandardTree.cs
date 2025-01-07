public class BaggedPredictorStandardTree : BaggedPredictor<StandardTree>
{
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public StandardTree.Reduction reduction;

    public BaggedPredictorStandardTree(List<Sample> samples, int minSamplesPerLeaf, int maxLeafDepth, StandardTree.Reduction reduction)
        : base(samples)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.reduction = reduction;
    }

    protected override StandardTree AddModel(List<Sample> baggedSamples)
    {
        return new StandardTree(baggedSamples, minSamplesPerLeaf, maxLeafDepth, reduction);
    }

    protected override float[] Predict(StandardTree model, float[] input)
    {
        return model.Predict(input);
    }
}