public class BaggedPredictorStandardTree : BaggedPredictor<StandardTree>
{
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public StandardTree.Reduction reduction;

    public BaggedPredictorStandardTree(List<Sample> samples, int featuresPerBag, int minSamplesPerLeaf, int maxLeafDepth, StandardTree.Reduction reduction)
        : base(samples, featuresPerBag)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.reduction = reduction;
    }

    protected override StandardTree AddModel(List<Sample> baggedSamples, List<int> baggedFeatures)
    {
        return new StandardTree(baggedSamples, baggedFeatures, minSamplesPerLeaf, maxLeafDepth, reduction);
    }

    protected override float[] Predict(StandardTree model, float[] input)
    {
        return model.Predict(input);
    }
}