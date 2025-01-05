public class WeightedRandomForest
{
    public int outputComponentCount;
    public int[] allXComponents;
    public int xComponentCount;
    public int minSamplesPerLeaf;
    public List<WeightedRandomTree> weightedRandomTrees;

    public WeightedRandomForest(List<WeightedSample> weightedSamples, int xComponentCount, int treeCount, int minSamplesPerLeaf, bool verbose)
    {
        // confirm we have weightedSamples to work with
        if (weightedSamples.Count == 0)
        {
            throw new ArgumentException("WeightedSamples must not be empty.", nameof(weightedSamples));
        }

        this.outputComponentCount = weightedSamples[0].output.Length;
        this.xComponentCount = xComponentCount;
        this.minSamplesPerLeaf = minSamplesPerLeaf;

        // initialize the random trees list
        this.weightedRandomTrees = new List<WeightedRandomTree>();

        // create an array of all the x components
        this.allXComponents = Enumerable.Range(0, weightedSamples[0].input.Length).ToArray();

        // create the number of trees needed
        for (int treeIndex = 0; treeIndex < treeCount; treeIndex++)
        {
            if (verbose)
            {
                Console.WriteLine($"Random Forest, Building Tree: {treeIndex + 1}/{treeCount}");
            }
            AddTree(weightedSamples);
        }
    }

    public void AddTree(List<WeightedSample> weightedSamples)
    {
        // create a local random instance (for thread safety)
        Random random = new Random();

        // create a list of random weightedSamples
        List<WeightedSample> randomWeightedSamples = new List<WeightedSample>();

        // randomly resample with replacement up to sample count
        for (int sampleIndex = 0; sampleIndex < weightedSamples.Count; sampleIndex++)
        {
            int randomIndex = random.Next(weightedSamples.Count);
            randomWeightedSamples.Add(weightedSamples[randomIndex]);
        }

        // randomly shuffle the x components and take the first xComponentCount
        List<int> randomXComponents = allXComponents.OrderBy(x => random.Next()).Take(xComponentCount).ToList();

        // create random tree
        WeightedRandomTree weightedRandomTree = new WeightedRandomTree(random, randomWeightedSamples, randomXComponents, minSamplesPerLeaf);
        weightedRandomTrees.Add(weightedRandomTree);
    }

    public float[] Predict(float[] input)
    {
        float[] average = new float[outputComponentCount];
        foreach(WeightedRandomTree weightedRandomTree in weightedRandomTrees)
        {
            float[] prediction = weightedRandomTree.Predict(input);
            for (int i = 0; i < outputComponentCount; i++)
            {
                average[i] += prediction[i];
            }
        }
        for (int i = 0; i < outputComponentCount; i++)
        {
            average[i] /= (float)weightedRandomTrees.Count;
        }
        return average;
    }

    public long CountNodes()
    {
        long count = 0;
        foreach(WeightedRandomTree weightedRandomTree in weightedRandomTrees)
        {
            weightedRandomTree.CountNodes(ref count);
        }
        return count;
    }
}