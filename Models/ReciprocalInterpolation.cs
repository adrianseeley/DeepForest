public class ReciprocalInterpolation : Model
{
    public float coefficient;
    public List<Sample> samples;
    public float[] distances;

    public ReciprocalInterpolation(float coefficient) 
    {
        this.coefficient = coefficient;
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

        Parallel.For(0, samples.Count, i =>
        {
            Sample sample = samples[i];
            float distance = Utility.EuclideanDistance(input, sample.input);
            distances[i] = distance;
        });
        float distanceMin = distances.Min();
        float distanceMax = distances.Max();
        float distanceRange = distanceMax - distanceMin;
        List<int> zeroIndices = new List<int>();
        object zeroIndicesLock = new object();
        Parallel.For(0, samples.Count, i =>
        {
            float distanceNormal = (distances[i] - distanceMin) / distanceRange;
            distances[i] = 1f / (coefficient * distanceNormal + 0.00000001f);
            if (distanceNormal == 0)
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