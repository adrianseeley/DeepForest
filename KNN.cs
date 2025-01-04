public class KNN
{
    public int k;
    public List<Sample> samples;
    public List<int> features;

    public KNN(int k, List<Sample> samples, List<int> features)
    {
        this.k = k;
        this.samples = samples;
        this.features = features;
    }

    public static float Distance(float[] a, float[] b, List<int> features)
    {
        float distance = 0f;
        foreach(int feature in features)
        {
            distance += MathF.Pow(a[feature] - b[feature], 2);
        }
        return MathF.Sqrt(distance);
    }

    public float[] Predict(float[] input, bool excludeZero)
    {
        List<(Sample sample, float distance)> sampleDistances = new List<(Sample sample, float distance)>();
        foreach(Sample sample in samples)
        {
            float distance = Distance(input, sample.input, features);
            if (excludeZero && distance == 0f)
            {
                continue;
            }
            sampleDistances.Add((sample, distance));
        }
        // if we have no samples distances, return a zero vector (this can happen for homogenous features and exclude zero)
        if (sampleDistances.Count == 0)
        {
            return new float[samples[0].output.Length];
        }
        sampleDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        float[] average = new float[samples[0].output.Length];
        int minK = Math.Min(k, samples.Count);
        for (int i = 0; i < minK; i++)
        {
            Sample sample = sampleDistances[i].sample;
            for (int j = 0; j < average.Length; j++)
            {
                average[j] += sample.output[j];
            }
        }
        for (int i = 0; i < average.Length; i++)
        {
            average[i] /= (float)minK;
        }
        return average;
    }
}
