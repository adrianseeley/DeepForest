public class ResidualRandomForest
{
    public int outputComponentCount;
    public int[] allXComponents;
    public int xComponentCount;
    public int minSamplesPerLeaf;
    public float learningRate;
    public List<RandomTree> randomTrees;

    public ResidualRandomForest(List<Sample> samples, int xComponentCount, int treeCount, int minSamplesPerLeaf, float learningRate, bool verbose)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        this.outputComponentCount = samples[0].output.Length;
        this.xComponentCount = xComponentCount;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.learningRate = learningRate;

        // initialize the random trees list
        this.randomTrees = new List<RandomTree>();

        // create an array of all the x components
        this.allXComponents = Enumerable.Range(0, samples[0].input.Length).ToArray();

        // create accumulators (zeroed)
        List<float[]> accumulators = new List<float[]>();
        foreach(Sample sample in samples)
        {
            accumulators.Add(new float[sample.output.Length]);
        }

        // create the number of trees needed
        for (int treeIndex = 0; treeIndex < treeCount; treeIndex++)
        {
            if (verbose)
            {
                Console.WriteLine($"Residual Tree {treeIndex + 1}/{treeCount}");
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

            // create a tree using the residual samples
            AddTree(residuals);

            // update the accumulators
            for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
            {
                Sample residualSample = residuals[sampleIndex];
                float[] prediction = randomTrees[treeIndex].Predict(residualSample.input);
                for (int i = 0; i < prediction.Length; i++)
                {
                    accumulators[sampleIndex][i] += learningRate * prediction[i];
                }
            }
        }
    }

    public void AddTree(List<Sample> samples)
    {
        // create a local random instance (for thread safety)
        Random random = new Random();

        // create a list of random samples
        List<Sample> randomSamples = new List<Sample>();

        // randomly resample with replacement up to sample count
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            int randomIndex = random.Next(samples.Count);
            randomSamples.Add(samples[randomIndex]);
        }

        // randomly shuffle the x components and take the first xComponentCount
        List<int> randomXComponents = allXComponents.OrderBy(x => random.Next()).Take(xComponentCount).ToList();

        // create random tree
        RandomTree randomTree = new RandomTree(random, randomSamples, randomXComponents, minSamplesPerLeaf);
        randomTrees.Add(randomTree);
    }

    public float[] Predict(float[] input)
    {
        // create an accumulator (zeroed)
        float[] accumulator = new float[outputComponentCount];

        // iterate through the random trees
        foreach(RandomTree randomTree in randomTrees)
        {
            // predict and accumulate
            float[] prediction = randomTree.Predict(input);
            for (int i = 0; i < prediction.Length; i++)
            {
                accumulator[i] += learningRate * prediction[i];
            }
        }

        // return the accumulator (prediction)
        return accumulator;
    }

    public long CountNodes()
    {
        long count = 0;
        foreach(RandomTree randomTree in randomTrees)
        {
            randomTree.CountNodes(ref count);
        }
        return count;
    }
}