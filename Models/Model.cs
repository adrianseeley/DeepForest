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

    public List<float[]> PredictSamples(List<Sample> samples, bool verbose = false)
    {
        List<float[]> predictions = new List<float[]>(samples.Count);
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            Sample sample = samples[sampleIndex];
            if (verbose)
            {
                Console.Write($"\rPredicting sample {sampleIndex + 1} of {samples.Count}");
            }
            predictions.Add(Predict(sample.input));
        }
        if (verbose)
        {
            Console.WriteLine();
        }
        return predictions;
    }
}