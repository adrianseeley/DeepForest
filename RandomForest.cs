public class RandomForest
{
    public int outputComponentCount;
    public int[] allXComponents;
    public int xComponentCount;
    public int minSamplesPerLeaf;
    public List<RandomTree> randomTrees;
    public object randomTreesLock;

    public RandomForest(List<Sample> samples, int xComponentCount, int treeCount, int minSamplesPerLeaf)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        this.outputComponentCount = samples[0].output.Length;
        this.xComponentCount = xComponentCount;
        this.minSamplesPerLeaf = minSamplesPerLeaf;

        // initialize the random trees list
        this.randomTrees = new List<RandomTree>();

        // create a lock for the random trees list
        this.randomTreesLock = new object();

        // create an array of all the x components
        this.allXComponents = Enumerable.Range(0, samples[0].input.Length).ToArray();

        // iterate through all the trees
        Parallel.For(0, treeCount, (treeIndex) =>
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

        // randomly shuffle the x components and take the first xComponentCount
        List<int> randomXComponents = allXComponents.OrderBy(x => random.Next()).Take(xComponentCount).ToList();

        // create random tree
        RandomTree randomTree = new RandomTree(random, randomSamples, randomXComponents, minSamplesPerLeaf);

        // lock the random trees list and add the random tree
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