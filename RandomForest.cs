public class RandomForest
{
    public Random random;
    public int yComponent;
    public List<RandomTree> randomTrees;

    public RandomForest(List<(List<float> input, List<float> output)> samples, int yComponent, int treeCount)
    {
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }
        this.random = new Random();
        this.yComponent = yComponent;
        randomTrees = new List<RandomTree>();
        int[] allXComponents = Enumerable.Range(0, samples[0].input.Count).ToArray();
        for (int i = 0; i < treeCount; i++)
        {
            List<(List<float> input, List<float> output)> randomSamples = new List<(List<float> input, List<float> output)>();
            for (int j = 0; j < samples.Count; j++)
            {
                int randomIndex = random.Next(samples.Count);
                randomSamples.Add(samples[randomIndex]);
            }
            int xComponentCount = random.Next(1, allXComponents.Length + 1);
            List<int> randomXComponents = allXComponents.OrderBy(x => random.Next()).Take(xComponentCount).ToList();
            randomTrees.Add(new RandomTree(randomSamples, randomXComponents, yComponent));
        }
    }

    public float Predict(List<float> input)
    {
        return randomTrees.Select(t => t.Predict(input)).Average();
    }
}