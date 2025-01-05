public class HeavyResidualRandomForest
{
    public int outputComponentCount;
    public int xComponentCount;
    public int treeCount;
    public int minSamplesPerLeaf;
    public float learningRate;
    public int threadCount;
    public List<RandomForest> randomForests;

    public HeavyResidualRandomForest(List<Sample> samples, int xComponentCount, int forestCount, int treeCount, int minSamplesPerLeaf, float learningRate, int threadCount, bool verbose)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        this.outputComponentCount = samples[0].output.Length;
        this.xComponentCount = xComponentCount;
        this.treeCount = treeCount;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.learningRate = learningRate;
        this.threadCount = threadCount;

        // initialize the random forests list
        this.randomForests = new List<RandomForest>();

        // create accumulators (zeroed)
        List<float[]> accumulators = new List<float[]>();
        foreach(Sample sample in samples)
        {
            accumulators.Add(new float[sample.output.Length]);
        }

        // create the number of forests needed
        for (int forestIndex = 0; forestIndex < forestCount; forestIndex++)
        {
            if (verbose)
            {
                Console.WriteLine($"Residual Forest {forestIndex + 1}/{forestCount}");
            }

            // create residual samples
            List<Sample> residuals = new List<Sample>();
            for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
            {
                Sample sample = samples[sampleIndex];
                float[] accumulator = accumulators[sampleIndex];
                float[] residual = new float[sample.output.Length];
                for (int i = 0; i < sample.output.Length; i++)
                {
                    residual[i] = sample.output[i] - accumulator[i];
                }
                residuals.Add(new Sample(sample.input, residual));
            }

            // create a forest using the residual samples
            AddForest(residuals);

            // update the accumulators
            for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
            {
                Sample residualSample = residuals[sampleIndex];
                float[] prediction = randomForests[forestIndex].Predict(residualSample.input);
                for (int i = 0; i < prediction.Length; i++)
                {
                    accumulators[sampleIndex][i] += learningRate * prediction[i];
                }
            }
        }
    }

    public void AddForest(List<Sample> samples)
    {
        // create a random forest
        RandomForest randomForest = new RandomForest(samples, xComponentCount, treeCount, minSamplesPerLeaf, threadCount);
        randomForests.Add(randomForest);
    }

    public float[] Predict(float[] input)
    {
        // create an accumulator (zeroed)
        float[] accumulator = new float[outputComponentCount];

        // iterate through the random forests
        foreach(RandomForest randomForest in randomForests)
        {
            // predict and accumulate
            float[] prediction = randomForest.Predict(input);
            for (int i = 0; i < prediction.Length; i++)
            {
                accumulator[i] += learningRate * prediction[i];
            }
        }

        // return the accumulator (prediction)
        return accumulator;
    }
}