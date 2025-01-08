public class ResidualPredictorKNN : ResidualPredictor<KNN>
{
    public int k;
    public KNN.DistanceDelegate distanceDelegate;

    public ResidualPredictorKNN(List<Sample> samples, float learningRate, int k, KNN.DistanceDelegate distanceDelegate) 
        : base(samples, learningRate)
    {
        this.k = k;
        this.distanceDelegate = distanceDelegate;
    }

    protected override KNN AddModel(List<Sample> residuals)
    {
        KNN model = new KNN(residuals, k, distanceDelegate);
        return model;
    }

    protected override float[] Predict(KNN model, float[] input)
    {
        return model.Predict(input);
    }
}