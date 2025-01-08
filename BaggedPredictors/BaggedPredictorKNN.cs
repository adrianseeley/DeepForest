public class BaggedPredictorKNN : BaggedPredictor<KNN>
{
    public int k;
    public KNN.DistanceDelegate distanceDelegate;

    public BaggedPredictorKNN(List<Sample> samples, int featuresPerBag, int k, KNN.DistanceDelegate distanceDelegate)
        : base(samples, featuresPerBag)
    {
        this.k = k;
        this.distanceDelegate = distanceDelegate;
    }

    protected override KNN AddModel(List<Sample> baggedSamples, List<int> baggedFeatures)
    {
        return new KNN(baggedSamples, baggedFeatures, k, distanceDelegate);
    }

    protected override float[] Predict(KNN model, float[] input)
    {
        return model.Predict(input);
    }
}