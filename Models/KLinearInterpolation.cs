public class KLinearInterpolation : Model
{
    public int k;
    public List<Sample> samples;
    public (float distance, Sample sample)[] distances;

    public KLinearInterpolation(int k) 
    {
        if (k < 2)
        {
            throw new ArgumentException("k must be greater than or equal to 2");
        }
        this.k = k;
        this.samples = new List<Sample>();
        this.distances = [];
    }

    public override void Fit(List<Sample> samples)
    {
        this.samples = samples;
        this.distances = new (float, Sample)[this.samples.Count];
    }

    public override float[] Predict(float[] input)
    {
        // ** only one thread at a time can call predict **

        Parallel.For(0, samples.Count, i =>
        {
            Sample sample = samples[i];
            float distance = Utility.EuclideanDistance(input, sample.input);
            distances[i] = (distance, sample);
        });

        (float distance, Sample sample)[] neighbours = distances.OrderBy(item => item.distance).Take(k).ToArray();
        float distanceMax = neighbours.Max(item => item.distance);
        float[] output = new float[samples[0].output.Length];
        float totalWeight = 0f;
        for (int i = 0; i < neighbours.Length; i++)
        {
            Sample sample = neighbours[i].sample;
            float weight = 1f - (neighbours[i].distance / distanceMax);
            weight = Math.Max(weight, 0.00001f);
            totalWeight += weight;
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += sample.output[j] * weight;
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= totalWeight;
        }
        return output;
    }
}