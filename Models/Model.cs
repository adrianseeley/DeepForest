public abstract class Model
{
    public abstract void Fit(List<Sample> samples, List<int> features);
    public abstract float[] Predict(float[] input);
}