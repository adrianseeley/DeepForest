public class RandomForest
{
    public Random random;
    public int yComponent;
    public List<RandomTree> randomTrees;

    public RandomForest(List<(List<float> input, List<float> output)> samples, int yComponent, int treeCount, bool extraRandom)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // create a local random instance (for thread safety)
        this.random = new Random();

        // mark the y component of interest
        this.yComponent = yComponent;

        // initialize the random trees list
        this.randomTrees = new List<RandomTree>();

        // create an array of all the x components
        int[] allXComponents = Enumerable.Range(0, samples[0].input.Count).ToArray();

        // iterate through all the trees
        for (int treeIndex = 0; treeIndex < treeCount; treeIndex++)
        {
            // create a list of random samples
            List<(List<float> input, List<float> output)> randomSamples = new List<(List<float> input, List<float> output)>();

            // randomly resample with replacement up to sample count
            for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
            {
                int randomIndex = random.Next(samples.Count);
                randomSamples.Add(samples[randomIndex]);
            }

            // randomly choose a number of x components from 1 to all
            int xComponentCount = random.Next(1, allXComponents.Length + 1);

            // randomly shuffle the x components and take the first xComponentCount
            List<int> randomXComponents = allXComponents.OrderBy(x => random.Next()).Take(xComponentCount).ToList();

            // add a new random tree to the list
            randomTrees.Add(new RandomTree(random, randomSamples, randomXComponents, yComponent, extraRandom));
        }
    }

    public float Predict(List<float> input)
    {
        // average the prediction of all the random trees
        return randomTrees.Select(t => t.Predict(input)).Average();
    }
}