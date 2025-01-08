public abstract class Model
{
    public static List<Model> Create(int count, Func<Model> factory)
    {
        List<Model> models = new List<Model>();
        for (int i = 0; i < count; i++)
        {
            models.Add(factory());
        }
        return models;
    }

    public abstract void Fit(List<Sample> samples);
    public abstract float[] Predict(float[] input);

    public List<float[]> PredictSamples(List<Sample> samples)
    {
        List<float[]> predictions = new List<float[]>(samples.Count);
        foreach (Sample sample in samples)
        {
            predictions.Add(Predict(sample.input));
        }
        return predictions;
    }
}