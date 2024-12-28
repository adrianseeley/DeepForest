public class RandomForest
{
    public int yComponent;
    public List<RandomTree> randomTrees;

    public RandomForest(List<(List<float> input, List<float> output)> samples, int yComponent, int xComponentCount, int treeCount, bool extraRandom)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // mark the y component of interest
        this.yComponent = yComponent;

        // initialize the random trees list
        this.randomTrees = new List<RandomTree>();

        // create a lock for the random trees list
        object randomTreesLock = new object();

        // create an array of all the x components
        int[] allXComponents = Enumerable.Range(0, samples[0].input.Count).ToArray();

        // iterate through all the trees
        Parallel.For(0, treeCount, (treeIndex) =>
        {
            // create a local random instance (for thread safety)
            Random random = new Random();

            // create a list of random samples
            List<(List<float> input, List<float> output)> randomSamples = new List<(List<float> input, List<float> output)>();

            // randomly resample with replacement up to sample count
            for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
            {
                int randomIndex = random.Next(samples.Count);
                randomSamples.Add(samples[randomIndex]);
            }

            // randomly choose a number of x components from 1 to all
            //int xComponentCount = random.Next(1, allXComponents.Length + 1);

            // randomly shuffle the x components and take the first xComponentCount
            List<int> randomXComponents = allXComponents.OrderBy(x => random.Next()).Take(xComponentCount).ToList();

            // create random tree
            RandomTree randomTree = new RandomTree(random, randomSamples, randomXComponents, yComponent, extraRandom);

            // lock the random trees list and add the random tree
            lock (randomTreesLock)
            {
                randomTrees.Add(randomTree);
            }
        });
    }

    public float Predict(List<float> input)
    {
        // average the prediction of all the random trees
        return randomTrees.Select(t => t.Predict(input)).Average();
    }
}