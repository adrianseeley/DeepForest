public class LinearInterpolation : Model
{
    public List<Sample> samples;
    public float[] distances;

    public LinearInterpolation() 
    {
        this.samples = new List<Sample>();
        this.distances = [];
    }

    public override void Fit(List<Sample> samples)
    {
        this.samples = samples;
        this.distances = new float[this.samples.Count];
    }

    public override float[] Predict(float[] input)
    {
        // ** only one thread at a time can call predict **

        List<int> zeroIndices = new List<int>();
        object zeroIndicesLock = new object();
        Parallel.For(0, samples.Count, i =>
        {
            Sample sample = samples[i];
            float distance = Utility.EuclideanDistance(input, sample.input);
            distances[i] = distance;
            if (distance == 0)
            {
                lock (zeroIndicesLock)
                {
                    zeroIndices.Add(i);
                }
            }
        });
        if (zeroIndices.Count > 1)
        {
            List<Sample> zeroSamples = new List<Sample>();
            foreach (int zeroIndex in zeroIndices)
            {
                zeroSamples.Add(samples[zeroIndex]);
            }
            float[] zeroSampleAverage = Sample.AverageOutput(zeroSamples);
            return zeroSampleAverage;
        }
        float distanceMax = distances.Max();
        Parallel.For(0, samples.Count, i =>
        {
            float weight = 1f - (distances[i] / distanceMax);
            if (weight == 0f)
            {
                weight = 0.00001f;
            }
            distances[i] = weight;
        });
        float[] output = new float[samples[0].output.Length];
        float weightSum = distances.Sum();
        for (int i = 0; i < samples.Count; i++)
        {
            Sample sample = samples[i];
            float weight = distances[i];
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += sample.output[j] * weight;
            }
        }
        for (int j = 0; j < output.Length; j++)
        {
            output[j] /= weightSum;
        }
        return output;
    }
}