public class KNN
{
    public delegate float DistanceDelegate(float[] a, float[] b, List<int> features);
    public List<Sample> samples;
    public List<int> features;
    public int k;
    public DistanceDelegate distanceDelegate;

    public KNN(List<Sample> samples, List<int> features, int k, DistanceDelegate distanceDelegate)
    {
        this.samples = samples;
        this.features = features;
        this.k = k;
        this.distanceDelegate = distanceDelegate;
    }

    public float[] Predict(float[] input)
    {
        List<(Sample sample, float distance)> sampleDistances = new List<(Sample sample, float distance)>();
        foreach(Sample sample in samples)
        {
            float distance = distanceDelegate(input, sample.input, features);
            sampleDistances.Add((sample, distance));
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
