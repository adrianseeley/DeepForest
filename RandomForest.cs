public class RandomForest
{
    public int outputComponentCount;
    public int minSamplesPerLeaf;
    public int splitAttempts;
    public float flipRate;
    public List<RandomTree> randomTrees;
    public object randomTreesLock;

    public RandomForest(List<Sample> samples, int treeCount, int minSamplesPerLeaf, int splitAttempts, float flipRate, int threadCount)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        this.outputComponentCount = samples[0].output.Length;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.splitAttempts = splitAttempts;
        this.flipRate = flipRate;

        // initialize the random trees list
        this.randomTrees = new List<RandomTree>();

        // create lock
        this.randomTreesLock = new object();

        // create the number of trees needed
        ParallelOptions parallelOptions = new ParallelOptions();
        parallelOptions.MaxDegreeOfParallelism = threadCount;
        Parallel.For(0, treeCount, parallelOptions, treeIndex =>
        { 
            AddTree(samples);
        });
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

        // create random tree
        RandomTree randomTree = new RandomTree(random, randomSamples, minSamplesPerLeaf, splitAttempts, flipRate);
        
        // thread safe add it
        lock (randomTreesLock)
        {
            randomTrees.Add(randomTree);
        }
    }

    public float[] Predict(float[] input)
    {
        float[] average = new float[outputComponentCount];
        foreach(RandomTree randomTree in randomTrees)
        {
            float[] prediction = randomTree.Predict(input);
            for (int i = 0; i < outputComponentCount; i++)
            {
                average[i] += prediction[i];
            }
        }
        for (int i = 0; i < outputComponentCount; i++)
        {
            average[i] /= (float)randomTrees.Count;
        }
        return average;
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